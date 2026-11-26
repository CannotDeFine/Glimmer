#include "runtime.h"
#include "profile_archive.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_pointer_protocol.h"
#include "../backend/cuda_execution_domain_fixture.h"

#include <chrono>
#include <bit>
#include <cerrno>
#include <poll.h>
#include <source_location>
#include <sstream>
#include <sys/socket.h>
#include <thread>

#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
using namespace glimmer::test::cuda_domain;

struct ClientContextSettings {
    unsigned int flags{};
    bool expects_shared_close{};
};

ExecutionWireReply sdk_reply(
    std::optional<ExecutionWireReply> reply,
    const std::source_location location = std::source_location::current()) {
    if (!reply || reply->status || reply->original_status != CUDA_SUCCESS ||
        reply->disposition != ExecutionWireDisposition::kComplete)
        throw std::runtime_error(
            "actual SDK result: status=" + (reply ? std::to_string(reply->status) : "none") +
            ", original=" +
            (reply && reply->original_status ? std::to_string(*reply->original_status) : "none") +
            ", disposition=" +
            (reply ? std::to_string(static_cast<unsigned int>(reply->disposition)) : "none") +
            ", line=" + std::to_string(location.line()));
    return std::move(*reply);
}
void send_byte(int socket, char value) {
    expect(::send(socket, &value, 1, MSG_NOSIGNAL) == 1, "send exact fixture barrier");
}
void wait_byte(int socket, char expected) {
    pollfd descriptor{socket, POLLIN, 0};
    expect(::poll(&descriptor, 1, 10000) == 1, "bounded fixture barrier");
    char value{};
    expect(::recv(socket, &value, 1, 0) == 1 && value == expected, "exact fixture phase");
}
void client_input(int barrier, const std::string& path, TaskPeerIdentity server,
                  ExecutionContextPurpose purpose, unsigned int commands,
                  bool changes_configuration, bool independent_streams, bool disconnects,
                  CUresult expected_error, std::string_view source_suffix = {},
                  bool disconnects_before_reply = false, bool stages_second_command = false,
                  bool holds_host_gate = false, bool is_asynchronous = false,
                  bool prefers_shared_cache = false, ClientContextSettings context_settings = {}) {
    wait_byte(barrier, 'C');
    ExecutionSocketClient client({path, server, 10000});
    std::vector<std::byte> bootstrap;
    if (purpose != ExecutionContextPurpose::kLegacyIsolated) {
        const auto encoded =
            encode_execution_context_bootstrap({purpose, context_settings.flags, 1});
        expect(encoded.has_value(), "cold context-purpose codec");
        bootstrap.assign(encoded->begin(), encoded->end());
    }
    sdk_reply(client.open(bootstrap));
    if (context_settings.expects_shared_close && context_settings.flags) {
        const auto flags = sdk_reply(client.exchange(ExecutionOpcode::kGetContextFlags));
        expect(flags.id == (context_settings.flags | CU_CTX_MAP_HOST),
               "typed member initializes and restores its exact effective logical flags");
    }
    ExecutionPayloadWriter source;
    expect(source.write_string(std::string(kSource) + std::string(source_suffix)),
           "exact original PTX identity");
    const auto module = sdk_reply(client.exchange(ExecutionOpcode::kLoadPtx, source.take()));
    ExecutionPayloadWriter lookup;
    expect(lookup.write_u64(module.id) && lookup.write_string("kernel"), "original kernel lookup");
    const auto function = sdk_reply(client.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
    expect(function.parameter_bytes == 12 && function.arguments.size() == 2,
           "only original ABI exposed to the unmodified launch caller");
    ExecutionPayloadWriter extent;
    const auto output_bytes = independent_streams ? 32 * commands : 32;
    expect(extent.write_u64(output_bytes), "complete original output extent");
    const auto output = sdk_reply(client.exchange(ExecutionOpcode::kAllocate, extent.take()));
    std::vector<std::uint64_t> streams(commands);
    if (independent_streams) {
        for (auto& stream : streams) {
            ExecutionPayloadWriter flags;
            expect(flags.write_u32(CU_STREAM_NON_BLOCKING), "independent public streams");
            stream = sdk_reply(client.exchange(ExecutionOpcode::kCreateStream, flags.take())).id;
        }
    }
    if (prefers_shared_cache) {
        ExecutionPayloadWriter cache;
        expect(cache.write_u32(static_cast<std::uint32_t>(CudaContextControl::kSetCache)) &&
                   cache.write_u32(CU_FUNC_CACHE_PREFER_SHARED) && cache.write_u64(0),
               "set a real distinct logical member configuration before physical work");
        sdk_reply(client.exchange(ExecutionOpcode::kDriverContextControl, cache.take()));
    }
    std::uint64_t host_gate{};
    if (holds_host_gate) {
        ExecutionPayloadWriter stream;
        expect(stream.write_u64(0), "gate the exact default source stream");
        const auto queued = client.exchange(ExecutionOpcode::kEnqueueHostFunction, stream.take());
        expect(queued && !queued->status && !queued->original_status && queued->id &&
                   queued->disposition == ExecutionWireDisposition::kComplete,
               "actual CPU callback boundary, not fabricated GPU permission");
        host_gate = queued->id;
    }
    send_byte(barrier, 'R');
    expect(::send(barrier, &output.address, sizeof(output.address), MSG_NOSIGNAL) ==
               sizeof(output.address),
           "fixture publishes its original output identity, never a native execution grant");
    wait_byte(barrier, 'G');
    std::array<std::byte, 12> parameters{};
    const std::uint32_t value = 7;
    std::memcpy(parameters.data(), &output.address, 8);
    std::memcpy(parameters.data() + 8, &value, 4);
    std::vector<std::uint64_t> accepted;
    for (unsigned int index = 0; index < commands; ++index) {
        if (stages_second_command && index == 1) {
            send_byte(barrier, 'W');
            wait_byte(barrier, 'N');
        }
        auto packed = parameters;
        if (independent_streams) {
            const auto address = output.address + CUdeviceptr{32} * index;
            std::memcpy(packed.data(), &address, 8);
        }
        ExecutionPayloadWriter launch_input;
        expect(launch_input.write_u64(function.id) && launch_input.write_u64(streams[index]),
               "original per-stream logical handles");
        for (const auto dimension : {8U, 1U, 1U, 32U, 1U, 1U})
            expect(launch_input.write_u32(dimension), "unchanged logical geometry");
        expect(launch_input.write_u32(0) && launch_input.write_blob(packed), "exact original ABI");
        if (is_asynchronous)
            expect(launch_input.write_u32(kExecutionAsyncLaunchMagic),
                   "explicit copied ASY1 admission");
        const auto submitted = client.submit(ExecutionOpcode::kLaunch, launch_input.take());
        expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && client.poll_for(0),
               "send original launches before awaiting any host result");
        accepted.push_back(submitted.sequence);
    }
    send_byte(barrier, 'P');
    if (is_asynchronous) parameters.fill(std::byte{0xA5});
    if (holds_host_gate) {
        const auto control = [&](ExecutionOpcode operation, std::span<const std::byte> payload) {
            const auto submitted = client.submit(operation, payload);
            expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
                   "callback control uses actual pipelined receipts, not exclusive exchange");
            const auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds{10};
            std::optional<ExecutionWireReply> result;
            while (!(result = client.take_reply(submitted.sequence)))
                expect(client.poll_for(1) && std::chrono::steady_clock::now() < timeout,
                       "bounded exact callback reply alongside retained kernel receipts");
            return result;
        };
        wait_byte(barrier, 'F');
        ExecutionPayloadWriter query;
        expect(query.write_u64(host_gate), "query only the owned callback boundary");
        const auto permission = control(ExecutionOpcode::kQueryHostFunction, query.take());
        expect(permission && !permission->status && !permission->original_status &&
                   permission->id == host_gate,
               ("real callback permission: status=" +
                std::to_string(permission ? permission->status : -1) + ", original=" +
                (permission && permission->original_status
                     ? std::to_string(*permission->original_status)
                     : "none") +
                ", id=" + std::to_string(permission ? permission->id : 0) +
                ", expected=" + std::to_string(host_gate))
                   .c_str());
        send_byte(barrier, 'Q');
        wait_byte(barrier, 'E');
        ExecutionPayloadWriter returned;
        expect(returned.write_u64(host_gate) && returned.write_u32(1),
               "callback returned normally");
        const auto finished = control(ExecutionOpcode::kFinishHostFunction, returned.take());
        expect(finished && !finished->status && !finished->original_status &&
                   finished->id == host_gate,
               "settle the actual CPU barrier without claiming GPU completion");
    }
    if (disconnects_before_reply) {
        wait_byte(barrier, 'X');
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    std::uint64_t first_sequence = holds_host_gate ? 0 : 1;
    std::vector<ExecutionAsyncFrontierKey> asynchronous_keys;
    for (std::size_t index = 0; index < accepted.size(); ++index) {
        std::optional<ExecutionWireReply> reply;
        while (!(reply = client.take_reply(accepted[index])))
            expect(client.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                   "bounded actual host reply");
        if (expected_error != CUDA_SUCCESS) {
            expect(reply && reply->status == expected_error && !reply->original_status,
                   "known profiling rejection is not a fabricated native launch result");
        } else {
            std::uint64_t actual{};
            if (is_asynchronous) {
                expect(
                    reply && reply->disposition == ExecutionWireDisposition::kAdmissionAccepted &&
                        !reply->status && reply->id && reply->address && !reply->original_status &&
                        !reply->parameter_bytes && reply->arguments.empty() && reply->data.empty(),
                    "reliable ASY1 receipt is not an SDK result or physical completion");
                actual = reply->id;
                asynchronous_keys.push_back({reply->address, reply->id});
            } else
                actual = sdk_reply(std::move(reply)).id;
            if (holds_host_gate && !index) first_sequence = actual;
            expect(first_sequence && actual == first_sequence + index,
                   "one SDK reply for each same logical command");
        }
    }
    send_byte(barrier, 'H');
    if (disconnects) {
        wait_byte(barrier, 'X');
        return;
    }
    if (changes_configuration) {
        wait_byte(barrier, 'M');
        ExecutionPayloadWriter control;
        expect(control.write_u32(static_cast<std::uint32_t>(CudaContextControl::kSetCache)) &&
                   control.write_u32(CU_FUNC_CACHE_PREFER_SHARED) && control.write_u64(0),
               "original requested context configuration");
        const auto submitted =
            client.submit(ExecutionOpcode::kDriverContextControl, control.take());
        expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && client.poll_for(0),
               "post original configuration request before the physical-stop barrier");
        send_byte(barrier, 'Q');
        std::optional<ExecutionWireReply> configured;
        while (!(configured = client.take_reply(submitted.sequence)))
            expect(client.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                   "bounded actual deferred configuration result");
        sdk_reply(std::move(configured));
        send_byte(barrier, 'U');
    }
    wait_byte(barrier, 'T');
    if (changes_configuration) {
        ExecutionPayloadWriter query;
        expect(query.write_u32(static_cast<std::uint32_t>(CudaContextControl::kGetCache)) &&
                   query.write_u32(0) && query.write_u64(0),
               "exact logical cache query");
        expect(
            sdk_reply(client.exchange(ExecutionOpcode::kDriverContextControl, query.take())).id ==
                CU_FUNC_CACHE_PREFER_SHARED,
            "deferred metadata query restores this member, not the sibling physical configuration");
    }
    ExecutionPayloadWriter scope;
    expect(scope.write_u32(0) && scope.write_u64(0), "logical context frontier");
    const auto frontier = client.exchange(ExecutionOpcode::kCaptureFrontier, scope.take());
    expect(frontier && !frontier->status && !frontier->original_status, "CPU-only exact frontier");
    ExecutionPayloadWriter id;
    expect(id.write_u64(frontier->id), "exact frontier identity");
    const auto frontier_bytes = id.take();
    while (true) {
        const auto observed = client.exchange(ExecutionOpcode::kQueryFrontier, frontier_bytes);
        expect(observed && !observed->original_status, "actual logical completion observation");
        if (!observed->status) break;
        if (expected_error != CUDA_SUCCESS && observed->status == expected_error) break;
        expect(
            observed->status == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline,
            "actual retained continuation, not original replay");
    }
    const auto released = client.exchange(ExecutionOpcode::kReleaseFrontier, frontier_bytes);
    expect(released && !released->status, "release metadata only");
    if (!asynchronous_keys.empty()) {
        const auto keys = encode_execution_async_frontier_batch(asynchronous_keys);
        expect(keys.has_value(), "encode every physically joined ASY1 observation");
        const auto retired = client.exchange(ExecutionOpcode::kReleaseAsyncFrontiers, *keys);
        expect(retired && !retired->status && !retired->original_status && retired->data == *keys,
               "release exactly the original ASY1 records after complete physical proof");
    }
    ExecutionPayloadWriter download_input;
    expect(download_input.write_u64(output.id) && download_input.write_u64(0) &&
               download_input.write_u64(output_bytes),
           "complete output readback");
    const auto downloaded =
        sdk_reply(client.exchange(ExecutionOpcode::kDownload, download_input.take()));
    expect(downloaded.data.size() == output_bytes, "all output values returned");
    std::vector<std::uint32_t> values(output_bytes / 4);
    std::memcpy(values.data(), downloaded.data.data(), output_bytes);
    expect(std::ranges::all_of(values,
                               [commands, independent_streams, expected_error](auto actual) {
                                   return actual == (expected_error != CUDA_SUCCESS ? 0
                                                     : independent_streams          ? 7
                                                                                    : 7 * commands);
                               }),
           "every non-idempotent original output is exact");
    const auto closed = client.exchange(ExecutionOpcode::kClose);
    expect(closed && !closed->status &&
               closed->disposition == ExecutionWireDisposition::kComplete &&
               closed->original_status == (purpose == ExecutionContextPurpose::kRuntimePrimary ||
                                                   context_settings.expects_shared_close
                                               ? std::optional<std::int32_t>{}
                                               : std::optional<std::int32_t>{CUDA_SUCCESS}),
           "member-only Close is composite cleanup, not a fabricated native context destroy");
}
void gemm_client_input(int barrier, const std::string& path, TaskPeerIdentity server,
                       CUresult expected_error) {
    wait_byte(barrier, 'C');
    ExecutionSocketClient client({path, server, 10000});
    const auto bootstrap =
        encode_execution_context_bootstrap({ExecutionContextPurpose::kRuntimePrimary, 0, 1});
    expect(bootstrap.has_value(), "ordinary shared GEMM context descriptor");
    sdk_reply(client.open(*bootstrap));
    const auto handle = sdk_reply(client.exchange(ExecutionOpcode::kCreateBlas));
    std::array<ExecutionWireReply, 3> allocations;
    const std::array<std::array<float, 3>, 3> initial{std::array{-9.0F, 2.0F, -7.0F},
                                                      std::array{-8.0F, 4.0F, -6.0F},
                                                      std::array{-5.0F, 6.0F, -3.0F}};
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        ExecutionPayloadWriter extent;
        expect(extent.write_u64(sizeof(initial[index])), "complete guarded matrix extent");
        allocations[index] = sdk_reply(client.exchange(ExecutionOpcode::kAllocate, extent.take()));
        ExecutionPayloadWriter upload_input;
        expect(upload_input.write_u64(allocations[index].id) && upload_input.write_u64(0) &&
                   upload_input.write_blob(std::as_bytes(std::span{initial[index]})),
               "initialize every matrix value and both guards");
        sdk_reply(client.exchange(ExecutionOpcode::kUpload, upload_input.take()));
    }
    // Complete initialization before the parent makes later fake GPU events
    // non-ready. This must not depend on another client's connection work.
    ExecutionPayloadWriter scope;
    expect(scope.write_u32(0) && scope.write_u64(0), "GEMM initialization frontier scope");
    const auto frontier = client.exchange(ExecutionOpcode::kCaptureFrontier, scope.take());
    expect(frontier && !frontier->status && !frontier->original_status,
           "capture actual GEMM upload completion");
    ExecutionPayloadWriter frontier_id;
    expect(frontier_id.write_u64(frontier->id), "GEMM initialization frontier identity");
    const auto frontier_bytes = frontier_id.take();
    const auto ready = client.exchange(ExecutionOpcode::kQueryFrontier, frontier_bytes);
    expect(ready && !ready->status && !ready->original_status,
           "GEMM uploads complete before ready");
    const auto released = client.exchange(ExecutionOpcode::kReleaseFrontier, frontier_bytes);
    expect(released && !released->status, "release GEMM initialization frontier");
    send_byte(barrier, 'R');
    const auto output = allocations[2].address + sizeof(float);
    expect(::send(barrier, &output, sizeof(output), MSG_NOSIGNAL) == sizeof(output),
           "publish only the unchanged GEMM output address");
    wait_byte(barrier, 'G');
    ExecutionPayloadWriter computation;
    expect(computation.write_u64(handle.id) && computation.write_u32(CUBLAS_OP_N) &&
               computation.write_u32(CUBLAS_OP_N),
           "actual owned GEMM handle and transpose modes");
    for (unsigned int index = 0; index < 6; ++index)
        expect(computation.write_u32(1), "original one-element matrix dimensions and strides");
    for (const auto& allocation : allocations)
        expect(computation.write_u64(allocation.id) && computation.write_u64(sizeof(float)),
               "original guarded matrix ranges");
    expect(computation.write_u32(0) && computation.write_u32(std::bit_cast<std::uint32_t>(2.0F)) &&
               computation.write_u32(std::bit_cast<std::uint32_t>(3.0F)),
           "host alpha and nonzero beta prevent an idempotent replay oracle");
    const auto submitted = client.submit(ExecutionOpcode::kSgemm, computation.take());
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && client.poll_for(0),
           "actual application GEMM enters the same authenticated execution queue");
    send_byte(barrier, 'P');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    std::optional<ExecutionWireReply> reply;
    while (!(reply = client.take_reply(submitted.sequence)))
        expect(client.poll_for(1) && std::chrono::steady_clock::now() < deadline,
               "bounded actual GEMM host result");
    if (expected_error == CUDA_SUCCESS) {
        expect(reply->status == CUDA_SUCCESS && !reply->original_status && reply->id == 1 &&
                   reply->disposition == ExecutionWireDisposition::kComputeReplacement,
               "GEMM uses an actual owned replacement, not original fallback or a PTX disguise");
    } else {
        expect(reply->status == expected_error && !reply->original_status &&
                   reply->disposition == ExecutionWireDisposition::kRejected,
               "strict isolated profiling rejects the unsupported catalog before computation");
    }
    send_byte(barrier, 'H');
    wait_byte(barrier, 'T');
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        ExecutionPayloadWriter download_input;
        expect(download_input.write_u64(allocations[index].id) && download_input.write_u64(0) &&
                   download_input.write_u64(sizeof(initial[index])),
               "read every matrix element and guard after real physical drain");
        const auto downloaded =
            sdk_reply(client.exchange(ExecutionOpcode::kDownload, download_input.take()));
        auto expected = initial[index];
        if (index == 2 && expected_error == CUDA_SUCCESS) expected[1] = 34.0F;
        expect(downloaded.data.size() == sizeof(expected) &&
                   std::ranges::equal(downloaded.data, std::as_bytes(std::span{expected})),
               "all non-idempotent GEMM outputs, inputs and guards remain exact");
    }
    ExecutionPayloadWriter destroy;
    expect(destroy.write_u64(handle.id), "destroy only the actual owned BLAS handle");
    sdk_reply(client.exchange(ExecutionOpcode::kDestroyBlas, destroy.take()));
    const auto closed = client.exchange(ExecutionOpcode::kClose);
    expect(closed && !closed->status && !closed->original_status &&
               closed->disposition == ExecutionWireDisposition::kComplete,
           "ordinary GEMM member retires without destroying its siblings' native domain");
}
void pointer_selection_client(int barrier, const std::string& path, TaskPeerIdentity server,
                              bool skips_driver_query) {
    wait_byte(barrier, 'C');
    const ExecutionSocketClientConfig options{path, server, 10000};
    ExecutionSocketClient primary(options), metadata(options), first(options), second(options),
        third(options), replacement(options);
    const auto open = [&](ExecutionSocketClient& client, ExecutionContextPurpose purpose,
                          std::uint64_t epoch = 1) {
        const auto bootstrap = encode_execution_context_bootstrap({purpose, 0, epoch});
        expect(bootstrap.has_value(), "actual cold generation descriptor");
        return sdk_reply(client.open(*bootstrap));
    };
    open(primary, ExecutionContextPurpose::kRuntimePrimary);
    const auto original_session = primary.session();
    ExecutionPayloadWriter extent;
    expect(extent.write_u64(32), "original primary allocation extent");
    const auto original = sdk_reply(primary.exchange(ExecutionOpcode::kAllocate, extent.take()));
    ExecutionPayloadWriter role;
    expect(role.write_u32(kExecutionPointerMetadataRole) && role.write_u32(0), "PMD1 helper role");
    const auto opened = metadata.open(role.take());
    expect(opened && !opened->status && !opened->original_status, "metadata owns no CUDA context");
    const auto query = [&](ExecutionPointerDomain domain) {
        ExecutionPayloadWriter input;
        expect(write_execution_pointer_query(
                   input, {domain, domain == ExecutionPointerDomain::kDriver ? 9U : 0U,
                           ExecutionPointerKind::kDevice, primary.session(), primary.session(),
                           original.id, 0}),
               "authenticated exact query/source generations");
        const auto reply = metadata.exchange(ExecutionOpcode::kPointerMetadata, input.take());
        if (domain == ExecutionPointerDomain::kDriver) {
            const auto result = sdk_reply(reply);
            ExecutionPayloadReader reader(result.data);
            ExecutionPointerObservation observation;
            expect(read_execution_pointer_observation(reader, observation) && reader.empty() &&
                       observation.effects == 2 && observation.device == 0,
                   "original Driver ordinal and exact output effects");
        } else {
            expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
                       reply->status == CUDA_ERROR_NOT_SUPPORTED && !reply->original_status &&
                       reply->data.empty() && !reply->parameter_bytes,
                   "post-selection version rejection is not a requested Runtime error");
        }
    };
    const auto close_user = [&](ExecutionSocketClient& client) {
        sdk_reply(client.exchange(ExecutionOpcode::kClose));
    };
    const auto cache_control = [&](ExecutionSocketClient& client, CudaContextControl operation,
                                   unsigned int preference = 0) {
        ExecutionPayloadWriter input;
        expect(input.write_u32(static_cast<std::uint32_t>(operation)) &&
                   input.write_u32(preference) && input.write_u64(0),
               "original cache control");
        return sdk_reply(client.exchange(ExecutionOpcode::kDriverContextControl, input.take()));
    };
    open(first, ExecutionContextPurpose::kExplicitUser);
    if (!skips_driver_query) query(ExecutionPointerDomain::kDriver);
    open(second, ExecutionContextPurpose::kExplicitUser);
    close_user(first);
    (void)cache_control(second, CudaContextControl::kGetCache);
    query(ExecutionPointerDomain::kRuntime);
    open(third, ExecutionContextPurpose::kExplicitUser);
    close_user(second);
    close_user(third);
    (void)cache_control(primary, CudaContextControl::kSetCache, CU_FUNC_CACHE_PREFER_SHARED);
    send_byte(barrier, 'R');
    expect(::send(barrier, &original.address, sizeof(original.address), MSG_NOSIGNAL) ==
               sizeof(original.address),
           "actual original allocation identity");
    wait_byte(barrier, 'G');
    const auto submitted = primary.submit(ExecutionOpcode::kClose);
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && primary.poll_for(0),
           "submit primary generation retirement without inventing physical completion");
    send_byte(barrier, 'Q');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    std::optional<ExecutionWireReply> closed;
    while (!(closed = primary.take_reply(submitted.sequence)))
        expect(primary.poll_for(1) && std::chrono::steady_clock::now() < deadline,
               "bounded deferred member-only generation retirement");
    expect(!closed->status && !closed->original_status &&
               closed->disposition == ExecutionWireDisposition::kComplete,
           "primary close never destroys or fabricates a return from the shared native context");
    open(replacement, ExecutionContextPurpose::kRuntimePrimary, 2);
    expect(
        replacement.session() != original_session &&
            cache_control(replacement, CudaContextControl::kGetCache).id ==
                CU_FUNC_CACHE_PREFER_NONE,
        "rejoined logical generation inherits original baseline, not previous or physical state");
    send_byte(barrier, 'U');
    ExecutionPayloadWriter source;
    expect(source.write_string(kSource), "unchanged original code after primary rejoin");
    const auto module = sdk_reply(replacement.exchange(ExecutionOpcode::kLoadPtx, source.take()));
    ExecutionPayloadWriter lookup;
    expect(lookup.write_u64(module.id) && lookup.write_string("kernel"), "fresh original function");
    const auto function =
        sdk_reply(replacement.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
    ExecutionPayloadWriter bytes;
    expect(bytes.write_u64(32), "fresh generation independent output");
    const auto output = sdk_reply(replacement.exchange(ExecutionOpcode::kAllocate, bytes.take()));
    std::array<std::byte, 12> parameters{};
    const std::uint32_t value = 7;
    std::memcpy(parameters.data(), &output.address, 8);
    std::memcpy(parameters.data() + 8, &value, 4);
    ExecutionPayloadWriter launch;
    expect(launch.write_u64(function.id) && launch.write_u64(0),
           "fresh generation logical handles");
    for (const auto dimension : {8U, 1U, 1U, 32U, 1U, 1U})
        expect(launch.write_u32(dimension), "original geometry");
    expect(launch.write_u32(0) && launch.write_blob(parameters),
           "original non-idempotent arguments");
    sdk_reply(replacement.exchange(ExecutionOpcode::kLaunch, launch.take()));
    ExecutionPayloadWriter download;
    expect(download.write_u64(output.id) && download.write_u64(0) && download.write_u64(32),
           "full fresh result");
    const auto result =
        sdk_reply(replacement.exchange(ExecutionOpcode::kDownload, download.take()));
    std::array<std::uint32_t, 8> values{};
    expect(result.data.size() == sizeof(values), "complete fresh output extent");
    std::memcpy(values.data(), result.data.data(), sizeof(values));
    expect(std::ranges::all_of(values, [](auto actual) { return actual == 7; }),
           "fresh generation executes exactly once while the sibling remains intact");
    const auto retired = replacement.exchange(ExecutionOpcode::kClose);
    expect(retired && !retired->status && !retired->original_status,
           "retire only the fresh member");
    const auto helper = metadata.exchange(ExecutionOpcode::kClose);
    expect(helper && !helper->status && !helper->original_status,
           "metadata helper has no native owner to destroy");
    send_byte(barrier, 'D');
}

struct Child {
    pid_t pid{-1};
    int socket{-1};
    bool ready{};
    CUdeviceptr output_address{};
    Child(const std::string& path, TaskPeerIdentity server, ExecutionContextPurpose purpose,
          unsigned int commands = 1, bool changes_configuration = false,
          bool independent_streams = false, bool disconnects = false,
          CUresult expected_error = CUDA_SUCCESS, unsigned int pointer_selection = 0,
          std::string_view source_suffix = {}, bool disconnects_before_reply = false,
          bool stages_second_command = false, bool executes_gemm = false,
          bool holds_host_gate = false, bool is_asynchronous = false,
          bool prefers_shared_cache = false, ClientContextSettings context_settings = {}) {
        std::array<int, 2> sockets{};
        expect(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets.data()) == 0,
               "fixture socketpair");
        pid = ::fork();
        expect(pid >= 0, "independent authenticated client");
        if (!pid) {
            (void)::close(sockets[0]);
            int status = EXIT_SUCCESS;
            try {
                if (pointer_selection)
                    pointer_selection_client(sockets[1], path, server, pointer_selection == 2);
                else if (executes_gemm)
                    gemm_client_input(sockets[1], path, server, expected_error);
                else
                    client_input(sockets[1], path, server, purpose, commands, changes_configuration,
                                 independent_streams, disconnects, expected_error, source_suffix,
                                 disconnects_before_reply, stages_second_command, holds_host_gate,
                                 is_asynchronous, prefers_shared_cache, context_settings);
            } catch (const std::exception& error) {
                std::cerr << "Pair client (" << path << "): " << error.what() << '\n';
                status = EXIT_FAILURE;
            }
            (void)::close(sockets[1]);
            std::exit(status);
        }
        (void)::close(sockets[1]);
        socket = sockets[0];
    }
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    ~Child() {
        if (socket >= 0) (void)::close(socket);
        if (pid > 0) {
            int status{};
            while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }
    TaskPeerIdentity identity() const {
        std::uint64_t start{};
        expect(read_process_start_time(static_cast<std::uint32_t>(pid), &start),
               "actual pinned child generation");
        return {static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()), start};
    }
    void observe_ready() {
        if (ready) return;
        char value{};
        const auto received = ::recv(socket, &value, 1, MSG_DONTWAIT);
        if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return;
        expect(received == 1 && value == 'R', "child original resources ready");
        expect(::recv(socket, &output_address, sizeof(output_address), MSG_WAITALL) ==
                   sizeof(output_address),
               "exact original output identity from owned fixture child");
        ready = true;
    }
    void join() {
        int status{};
        expect(::waitpid(pid, &status, 0) == pid, "join exact owned child");
        pid = -1;
        expect(WIFEXITED(status) && WEXITSTATUS(status) == 0, "child full numerical/status checks");
    }
};

void write_measurements(ExecutionServiceConfig& config, bool is_cold) {
    auto singles = pair_singles();
    const auto identity = profile_runtime_identity();
    expect(identity.has_value(), "exact program profile identity");
    const auto single_archive = singles->encode(*identity);
    expect(single_archive && write_profile_archive(config.profile_input, *single_archive),
           "protected readonly exact original archive");
    if (config.paired->profile_input.empty()) return;
    CudaSessionPairKernelDescription kernel;
    kernel.profile = singles->bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                                   kSource, {}, "kernel");
    kernel.geometry = {{8, 1, 1}, {32, 1, 1}, 0};
    kernel.limits = {80, 32, 2048, 65536, 320, 98304};
    kernel.variant = {glimmer::transform::PtxExecutionKind::kPreemptive, 320, 17, 4, 128};
    const auto pairs = pair_measurements(singles, {kernel, kernel}, is_cold);
    const auto pair_archive = pairs->encode(*identity);
    expect(pair_archive && write_execution_archive(config.paired->profile_input, *pair_archive,
                                                   kCudaPairProfileArchiveMaxBytes),
           "protected readonly exact pair archive");
}

void write_global_measurements(ExecutionServiceConfig& config, bool is_cold, bool ties) {
    auto single_config = config.profile;
    CudaProfileCache learning(single_config, pair_clock);
    std::array<std::string, 3> sources;
    for (std::size_t index = 0; index < sources.size(); ++index) {
        sources[index] = std::string(kSource) + "\n// candidate " + std::to_string(index) + "\n";
        const auto function =
            learning.bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                          sources[index], {}, "kernel");
        const std::array<unsigned int, 3> original_durations{200, 300, 100};
        for (const auto duration : {is_cold ? original_durations[index] : 100U, 80U}) {
            auto selected = learning.choose(function, {{8, 1, 1}, {32, 1, 1}, 0}, 320, true);
            expect(selected.observation != nullptr, "train independent exact original identity");
            pair_now_ns = 1000;
            selected.observation->begin();
            pair_now_ns += duration;
            selected.observation->finish(true);
        }
    }
    const auto identity = profile_runtime_identity();
    expect(identity.has_value(), "global profile runtime identity");
    const auto archive = learning.encode(*identity);
    auto singles = std::make_shared<CudaProfileCache>(single_config, pair_clock, false);
    expect(archive && singles->restore(*archive, *identity) &&
               write_profile_archive(config.profile_input, *archive),
           "readonly original profiles retain three distinct full source identities");
    CudaPairProfileCache pairs(singles, config.paired->profile, pair_clock);
    const CudaPairProfileLimits resources{80, 32, 2048, 65536, 320, 98304};
    std::array<CudaPairProfileCandidate, 4> candidates;
    std::size_t candidate{};
    for (const auto left : {80U, 160U})
        for (const auto right : {80U, 160U}) {
            candidates[candidate++].legs = {
                CudaPairProfileVariant{glimmer::transform::PtxExecutionKind::kPreemptive, left, 17,
                                       4, 128},
                CudaPairProfileVariant{glimmer::transform::PtxExecutionKind::kPreemptive, right, 17,
                                       4, 128}};
        }
    if (!is_cold)
        for (std::size_t first = 0; first < 3; ++first)
            for (std::size_t second = first + 1; second < 3; ++second) {
                std::array<CudaPairProfileKernel, 2> kernels;
                for (std::size_t leg = 0; leg < kernels.size(); ++leg) {
                    const auto index = leg ? second : first;
                    const auto function =
                        singles->bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                                      sources[index], {}, "kernel");
                    kernels[leg] = pairs.bind(function, {{8, 1, 1}, {32, 1, 1}, 0});
                    expect(static_cast<bool>(kernels[leg]), "exact bounded pair original binding");
                }
                for (std::size_t index = 0; index < candidates.size(); ++index) {
                    auto sampled =
                        pairs.reserve(kernels, {CudaPairProfileDomainKind::kSameContext, {99, 99}},
                                      resources, candidates);
                    expect(sampled && sampled.candidate == index,
                           "every exact joint catalog trains");
                    const auto duration =
                        index == 3 ? (!ties && first == 1 ? 40U : 100U) : 400U - 100U * index;
                    pair_now_ns = 1000;
                    sampled.observations[0]->begin();
                    pair_now_ns = 1005;
                    sampled.observations[1]->begin();
                    pair_now_ns = 1000 + duration;
                    sampled.observations[0]->finish(true);
                    pair_now_ns = 1005 + duration;
                    sampled.observations[1]->finish(true);
                }
            }
    const auto joint = pairs.encode(*identity);
    expect(joint && write_execution_archive(config.paired->profile_input, *joint,
                                            kCudaPairProfileArchiveMaxBytes),
           "publish exact joint profiles, never application replay");
}

unsigned int global_metadata_queries{};
unsigned int global_metadata_failures{};
CUresult global_metadata_status{CUDA_SUCCESS};
CUresult CUDAAPI global_device_attribute(int* output, CUdevice_attribute attribute,
                                         CUdevice device) {
    if (attribute == CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR) {
        ++global_metadata_queries;
        const auto status = std::exchange(global_metadata_status, CUDA_SUCCESS);
        if (status != CUDA_SUCCESS) {
            ++global_metadata_failures;
            return status;
        }
    }
    return pair_dispatch().device_attribute(output, attribute, device);
}
void global_service_pairs(const std::string& path, unsigned int scenario,
                          bool is_workload_aware = false) {
    driver = {};
    const bool starts_solo = scenario == 2 || scenario == 5 || scenario == 9;
    const bool starts_pair = scenario == 7 || scenario == 8;
    const bool cancels_trigger = scenario == 8;
    const auto server = current_process_identity();
    expect(server.has_value(), "actual global pair service owner");
    std::array<std::unique_ptr<Child>, 3> children;
    for (std::size_t index = 0; index < children.size(); ++index) {
        const auto suffix = "\n// candidate " + std::to_string(index) + "\n";
        children[index] = std::make_unique<Child>(
            path, *server, ExecutionContextPurpose::kRuntimePrimary, 1, false, false, false,
            CUDA_SUCCESS, 0, suffix, cancels_trigger && index == 2);
    }
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        configuration_domain_dispatch(), CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 4, 512});
    expect(domain->initialize() == CUDA_SUCCESS, "one real fake-SDK common domain");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 4;
    config.transport.max_pending_requests = 4;
    config.session_limits = limits(1000);
    config.persistent.workers = 320;
    for (std::size_t index = 0; index < children.size(); ++index)
        config.client_priorities.push_back({children[index]->identity(),
                                            scenario == 2 && index ? 100U : 0U,
                                            ExecutionClientKernelMode::kConfigured});
    config.profile.workers = {320};
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.profile_input = path + ".single";
    config.paired = ExecutionServicePairConfig{
        domain,
        {.max_candidates = 4, .warmup_calls = 0, .samples_per_candidate = 1},
        path + ".pair",
        {80, 160}};
    if (scenario == 4) config.paired->profile.max_pairs = 1;
    config.paired->should_rebalance_on_arrival =
        is_workload_aware || (scenario >= 5 && scenario != 9);
    if (is_workload_aware)
        config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kWorkloadAware;
    config.exit_when_idle = true;
    write_global_measurements(config, scenario == 1, scenario == 3);
    auto dispatch = pair_dispatch();
    dispatch.device_attribute = global_device_attribute;
    ExecutionService service(config, dispatch);
    expect(service.start(), "authenticated global pair service");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    for (auto& child : children) {
        send_byte(child->socket, 'C');
        while (!child->ready) {
            expect(service.step(1), "bootstrap members in a known connection order");
            child->observe_ready();
            expect(std::chrono::steady_clock::now() < deadline, "bounded global member setup");
        }
    }
    std::vector<CUdeviceptr> outputs;
    global_metadata_queries = 0;
    global_metadata_failures = 0;
    global_metadata_status = scenario == 10   ? CUDA_ERROR_NOT_SUPPORTED
                             : scenario == 11 ? CUDA_ERROR_INVALID_VALUE
                                              : CUDA_SUCCESS;
    driver.on_launch = [&] { outputs.push_back(driver.launch_output); };
    driver.query_status = CUDA_ERROR_NOT_READY;
    const std::size_t initial_count = starts_pair ? 2 : starts_solo ? 1 : 0;
    if (initial_count) {
        for (std::size_t index = 0; index < initial_count; ++index) {
            send_byte(children[index]->socket, 'G');
            wait_byte(children[index]->socket, 'P');
        }
        while (driver.launches < initial_count)
            expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
                   "initial work issues before actual late arrival");
        global_metadata_queries = 0;
    }
    for (std::size_t index = initial_count; index < children.size(); ++index) {
        send_byte(children[index]->socket, 'G');
        wait_byte(children[index]->socket, 'P');
    }
    for (unsigned int turn = 0; turn < 20; ++turn)
        expect(service.step(1), "ingest all candidates while physical events remain held");
    std::array<std::uint32_t, 2> arrival_retreat{};
    if (scenario >= 5)
        for (std::size_t index = 0; index < arrival_retreat.size(); ++index)
            std::memcpy(&arrival_retreat[index], driver.hosts[index].data.data(),
                        sizeof(std::uint32_t));
    if (cancels_trigger) {
        send_byte(children[2]->socket, 'X');
        children[2]->join();
        for (unsigned int turn = 0; turn < 20; ++turn)
            expect(service.step(1),
                   "observe trigger cancellation without releasing physical owners");
    }
    bool retained_first_stop = true;
    if (initial_count) {
        expect(driver.launches == initial_count, "no replacement before actual physical stop");
        driver.ready_launch = 1;
        for (unsigned int turn = 0; turn < 20; ++turn)
            expect(service.step(1), "observe exact first physical stop");
        if (starts_pair) {
            retained_first_stop =
                driver.launches == 2 && driver.counter_initializations == 2 && !driver.host_frees;
            driver.ready_launch = 2;
            for (unsigned int turn = 0; turn < 20; ++turn)
                expect(service.step(1), "reselect only after BOTH original physical stops");
        }
    }
    const auto first = initial_count;
    const std::array<std::size_t, 2> desired = scenario == 3 || cancels_trigger
                                                   ? std::array<std::size_t, 2>{0, 1}
                                                   : std::array<std::size_t, 2>{1, 2};
    const bool selected_expected =
        scenario >= 10 ? outputs.size() == 1 && outputs[0] == children[0]->output_address &&
                             driver.physical_workers == std::vector<unsigned int>{320} &&
                             global_metadata_failures == 1
        : scenario == 1 ? outputs.size() == 1 && outputs[0] == children[2]->output_address &&
                              driver.physical_workers == std::vector<unsigned int>{8}
                        : outputs.size() == first + 2 &&
                              outputs[first] == children[desired[0]]->output_address &&
                              outputs[first + 1] == children[desired[1]]->output_address &&
                              driver.physical_workers[first] == 160 &&
                              driver.physical_workers[first + 1] == 160;
    const bool bounded_metadata = global_metadata_queries <= 5;
    bool retained_counters = true;
    if (starts_pair)
        retained_counters = driver.previous_counters.size() >= 4 &&
                            driver.previous_counters[0] == 0 && driver.previous_counters[1] == 0 &&
                            driver.previous_counters[2] == 3 &&
                            driver.previous_counters[3] == (cancels_trigger ? 3 : 0);
    bool no_repeat_retreat = true;
    if (scenario >= 5 && scenario != 9) {
        for (const auto index : desired) {
            std::uint32_t word{};
            std::memcpy(&word, driver.hosts[index].data.data(), sizeof(word));
            no_repeat_retreat = no_repeat_retreat && !word;
        }
    }
    // Complete all original outputs and owned cleanup even for the OLD-runtime
    // negative control; a selection mismatch must not strand fixture children.
    driver.on_launch = {};
    driver.query_status = CUDA_SUCCESS;
    driver.ready_launch = 0;
    driver.next_claimed = 8;
    for (std::size_t index = 0; index < children.size(); ++index)
        if (!cancels_trigger || index != 2) send_byte(children[index]->socket, 'T');
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "all three unchanged non-idempotent commands complete and retire");
    for (std::size_t index = 0; index < children.size(); ++index) {
        if (cancels_trigger && index == 2) continue;
        wait_byte(children[index]->socket, 'H');
        children[index]->join();
    }
    expect(service.request_stop() && domain->snapshot().is_shutdown &&
               !domain->snapshot().is_quarantined && driver.live_resources() == 0 &&
               driver.context_creates == 1 && driver.context_destroys == 1,
           "global selection preserves complete normal child/resource/domain cleanup");
    expect(::unlink(config.profile_input.c_str()) == 0 &&
               ::unlink(config.paired->profile_input.c_str()) == 0,
           "remove only idle reproducible owned profile inputs");
    expect(selected_expected,
           "Global pair selection ignored a better or priority-eligible later candidate");
    expect(bounded_metadata,
           "global SDK metadata preparation must be linear in candidates, not pairs");
    if (scenario == 5 || starts_pair)
        expect(arrival_retreat[0] == 1 && (!starts_pair || arrival_retreat[1] == 1),
               "same-priority arrival did not request retreat before the actual physical stop");
    if (scenario == 6 || scenario == 9)
        expect(arrival_retreat[0] == 0 && arrival_retreat[1] == 0,
               "already considered co-arrivals and disabled policy cannot request retreat");
    expect(retained_first_stop && retained_counters && no_repeat_retreat,
           "checkpoint requires both real stops, retained counters and no repeated arrival churn");
}

constexpr std::array<std::string_view, 25> kMixedGemmNames{
    "glimmer_owned_sgemm_nn",        "glimmer_owned_sgemm_nt",   "glimmer_owned_sgemm_tn",
    "glimmer_owned_sgemm_tt",        "glimmer_owned_hgemm_nn",   "glimmer_owned_hgemm_nt",
    "glimmer_owned_hgemm_tn",        "glimmer_owned_hgemm_tt",   "glimmer_owned_hsgemm_nn",
    "glimmer_owned_hsgemm_nt",       "glimmer_owned_hsgemm_tn",  "glimmer_owned_hsgemm_tt",
    "glimmer_owned_bsgemm_nn",       "glimmer_owned_bsgemm_nt",  "glimmer_owned_bsgemm_tn",
    "glimmer_owned_bsgemm_tt",       "glimmer_owned_bhgemm_nn",  "glimmer_owned_bhgemm_nt",
    "glimmer_owned_bhgemm_tn",       "glimmer_owned_bhgemm_tt",  "glimmer_owned_bhsgemm_nn",
    "glimmer_owned_bhsgemm_nt",      "glimmer_owned_bhsgemm_tn", "glimmer_owned_bhsgemm_tt",
    "glimmer_owned_sgemm_nn_split_k"};
struct MixedGemmState {
    std::array<FakeFunction, kMixedGemmNames.size()> functions{};
    cudaStream_t stream{};
    bool is_live{};
    unsigned int lookups{}, launches{}, native_calls{}, creates{}, destroys{};
} mixed_gemm;

std::optional<std::size_t> mixed_gemm_function(CUfunction function) {
    for (std::size_t index = 0; index < mixed_gemm.functions.size(); ++index)
        if (function == reinterpret_cast<CUfunction>(&mixed_gemm.functions[index])) return index;
    return std::nullopt;
}
template <typename T>
T mixed_parameter(const std::byte* bytes, std::size_t offset) {
    T result{};
    std::memcpy(&result, bytes + offset, sizeof(result));
    return result;
}
std::thread::id sgemm_range_owner;
unsigned int sgemm_range_pushes{}, sgemm_range_pops{};
bool sgemm_range_expect_active{};

CudaSessionDispatch mixed_pair_dispatch() {
    auto dispatch = pair_dispatch();
    dispatch.device_attribute = global_device_attribute;
    dispatch.function_get = [](CUfunction* output, CUmodule module, const char* name) {
        const auto found = std::ranges::find(kMixedGemmNames, std::string_view{name});
        if (found == kMixedGemmNames.end()) return function_get(output, module, name);
        const auto* native = reinterpret_cast<FakeModule*>(module);
        expect(native && native->is_live && native->context == driver.current,
               "owned GEMM entries belong to their actual live selected-context provider module");
        const auto index = static_cast<std::size_t>(found - kMixedGemmNames.begin());
        mixed_gemm.functions[index].is_private = true;
        *output = reinterpret_cast<CUfunction>(&mixed_gemm.functions[index]);
        ++mixed_gemm.lookups;
        return CUDA_SUCCESS;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected CUDA/cuBLAS ABIs.
    dispatch.parameter_info = [](CUfunction function, std::size_t index, std::size_t* offset,
                                 std::size_t* bytes) {
        const auto variant = mixed_gemm_function(function);
        if (!variant) return parameter_info(function, index, offset, bytes);
        const bool is_split = *variant == 24;
        const auto arity = is_split ? std::size_t{17} : *variant < 12 ? 16 : 20;
        constexpr std::array<std::size_t, 20> offsets{0,  8,  16, 24, 28, 32, 36, 40,  44,  48,
                                                      52, 56, 64, 72, 80, 88, 96, 104, 112, 120};
        constexpr std::array<std::size_t, 20> widths{8, 8, 8, 4, 4, 4, 4, 4, 4, 4,
                                                     4, 8, 8, 4, 8, 8, 8, 8, 8, 4};
        if (index >= arity) return CUDA_ERROR_INVALID_VALUE;
        *offset = offsets[index];
        *bytes = is_split && index == 16 ? 4 : widths[index];
        return CUDA_SUCCESS;
    };
    dispatch.blas.create = [](cublasHandle_t* output) {
        expect(!mixed_gemm.is_live, "one owned ordinary BLAS handle");
        mixed_gemm.is_live = true;
        ++mixed_gemm.creates;
        *output = reinterpret_cast<cublasHandle_t>(&mixed_gemm);
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.destroy = [](cublasHandle_t handle) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&mixed_gemm) && mixed_gemm.is_live,
               "the actual native BLAS handle is destroyed exactly once");
        mixed_gemm.is_live = false;
        ++mixed_gemm.destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.set_stream = [](cublasHandle_t handle, cudaStream_t stream) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&mixed_gemm) && mixed_gemm.is_live,
               "configure only the real live BLAS handle");
        mixed_gemm.stream = stream;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.get_stream = [](cublasHandle_t handle, cudaStream_t* stream) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&mixed_gemm) && mixed_gemm.is_live,
               "observe the actual proxy stream, not a synthetic lookup result");
        *stream = mixed_gemm.stream;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                             const float*, const float*, int, const float*, int, const float*,
                             float*, int) {
        ++mixed_gemm.native_calls;
        return CUBLAS_STATUS_NOT_SUPPORTED;
    };
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int x, unsigned int y,
                                          unsigned int z, unsigned int bx, unsigned int by,
                                          unsigned int bz, unsigned int shared, CUstream stream,
                                          void** arguments, void** extra) {
        if (!mixed_gemm_function(function))
            return launch(function, x, y, z, bx, by, bz, shared, stream, arguments, extra);
        if (sgemm_range_expect_active)
            expect(sgemm_range_pushes == 1 && sgemm_range_pops == 0,
                   "the actual private GEMM launch occurs inside the owner range");
        expect(mixed_gemm_function(function) == 0 && x == 320 && y == 1 && z == 1 && bx == 128 &&
                   by == 1 && bz == 1 && !arguments && extra &&
                   *static_cast<const std::size_t*>(extra[3]) == 96,
               "actual private SGEMM preserves explicit320 workers and its independent96-byte ABI");
        const auto* bytes = static_cast<const std::byte*>(extra[1]);
        for (const auto offset : {24U, 28U, 32U, 36U, 40U, 44U})
            expect(mixed_parameter<std::uint32_t>(bytes, offset) == 1,
                   "selected GEMM dimensions match the unchanged public request");
        const auto output = mixed_parameter<CUdeviceptr>(bytes, 16);
        const auto counter = mixed_parameter<CUdeviceptr>(bytes, 80);
        const auto previous = mixed_parameter<std::uint64_t>(device_bytes(counter, 8), 0);
        expect(previous == 0 && !mixed_gemm.launches,
               "this complete ordinary GEMM cannot reset its counter or replay its beta update");
        const auto a =
            mixed_parameter<float>(device_bytes(mixed_parameter<CUdeviceptr>(bytes, 0), 4), 0);
        const auto b =
            mixed_parameter<float>(device_bytes(mixed_parameter<CUdeviceptr>(bytes, 8), 4), 0);
        const auto result =
            mixed_parameter<float>(bytes, 48) * a * b +
            mixed_parameter<float>(bytes, 52) * mixed_parameter<float>(device_bytes(output, 4), 0);
        std::memcpy(device_bytes(output, 4), &result, sizeof(result));
        const std::uint64_t claimed = 1;
        std::memcpy(device_bytes(counter, 8), &claimed, sizeof(claimed));
        driver.launch_output = output;
        if (driver.on_launch) driver.on_launch();
        ++driver.launches;
        ++mixed_gemm.launches;
        driver.physical_workers.push_back(x);
        return CUDA_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return dispatch;
}

int sgemm_range_push(const char* label) {
    expect(std::this_thread::get_id() == sgemm_range_owner &&
               std::string_view(label) == "glimmer.sgemm.target.issue.v1",
           "SGEMM range is pushed on the exact service owner thread");
    ++sgemm_range_pushes;
    return 0;
}
int sgemm_range_pop() {
    expect(std::this_thread::get_id() == sgemm_range_owner,
           "SGEMM range is popped on the exact service owner thread");
    ++sgemm_range_pops;
    return 0;
}

struct SgemmAuditCapture {
    std::ostringstream text;
    std::shared_ptr<spdlog::logger> logger{spdlog::default_logger()};
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink{
        std::make_shared<spdlog::sinks::ostream_sink_mt>(text)};
    SgemmAuditCapture() {
        sink->set_pattern("%v");
        logger->sinks().push_back(sink);
    }
    ~SgemmAuditCapture() {
        std::erase(logger->sinks(), sink);
    }
    SgemmAuditCapture(const SgemmAuditCapture&) = delete;
    SgemmAuditCapture& operator=(const SgemmAuditCapture&) = delete;
};

void mixed_service_pairs(const std::string& path, bool is_isolated, bool is_high_gemm = false,
                         bool misses_target = false, bool client_bound = false) {
    driver = {};
    mixed_gemm = {};
    sgemm_range_expect_active = is_high_gemm && !misses_target;
    global_metadata_queries = global_metadata_failures = 0;
    global_metadata_status = CUDA_SUCCESS;
    const auto server = current_process_identity();
    expect(server.has_value(), "actual authenticated mixed-command service owner");
    const auto count = is_isolated ? std::size_t{2} : 3;
    std::array<std::unique_ptr<Child>, 3> children;
    const auto expected_error = is_isolated ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
    for (std::size_t index = 0; index < count; ++index)
        children[index] = std::make_unique<Child>(
            path, *server, ExecutionContextPurpose::kRuntimePrimary, 1, false, false, false,
            expected_error, 0,
            is_isolated ? std::string{} : "\n// candidate " + std::to_string(index) + "\n", false,
            false, index == 0);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        configuration_domain_dispatch(), CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 4, 512});
    expect(domain->initialize() == CUDA_SUCCESS,
           "one shared native domain for public PTX and GEMM");
    ExecutionServiceConfig config;
    std::optional<SgemmAuditCapture> audit;
    if (is_high_gemm) {
        audit.emplace();
        config.should_audit_native = true;
        config.sgemm_audit_target = ExecutionSgemmAuditTarget{1, misses_target ? 4U : 1U};
        if (client_bound) {
            config.sgemm_audit_target->session = 0;
            config.sgemm_audit_target->client = children[0]->identity();
        }
        config.sgemm_range_hooks = ExecutionSgemmRangeHooks{sgemm_range_push, sgemm_range_pop};
        sgemm_range_owner = std::this_thread::get_id();
        sgemm_range_pushes = sgemm_range_pops = 0;
    }
    config.transport.socket_path = path;
    config.transport.max_peers = 4;
    config.transport.max_pending_requests = 4;
    config.session_limits = limits(1000);
    config.persistent.workers = 320;
    config.persistent.replace_sgemm = true;
    for (std::size_t index = 0; index < count; ++index)
        config.client_priorities.push_back({children[index]->identity(),
                                            is_high_gemm && index == 0 ? 100U : 0U,
                                            ExecutionClientKernelMode::kConfigured});
    config.profile.workers = {320};
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.profile_input = path + ".single";
    config.paired = ExecutionServicePairConfig{
        domain,
        {.max_candidates = 4, .warmup_calls = 0, .samples_per_candidate = 1},
        is_isolated ? std::string{} : path + ".pair",
        {80, 160},
        is_isolated,
        is_isolated ? path + ".measured" : std::string{}};
    config.exit_when_idle = true;
    if (is_isolated)
        write_measurements(config, false);
    else
        write_global_measurements(config, false, false);
    ExecutionService service(config, mixed_pair_dispatch());
    expect(service.start(), "same service configuration accepts all genuine application paths");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    const auto advance = [&] {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded mixed-command service progress");
    };
    for (std::size_t index = 0; index < count; ++index) {
        auto& child = children[client_bound ? (index + 1) % count : index];
        send_byte(child->socket, 'C');
        while (!child->ready) {
            advance();
            child->observe_ready();
        }
    }
    std::vector<CUdeviceptr> outputs;
    driver.on_launch = [&] { outputs.push_back(driver.launch_output); };
    driver.query_status = CUDA_ERROR_NOT_READY;
    driver.next_claimed = 8;
    global_metadata_queries = 0;
    for (std::size_t index = 0; index < count; ++index) {
        send_byte(children[index]->socket, 'G');
        wait_byte(children[index]->socket, 'P');
    }
    for (unsigned int turn = 0; turn < 20; ++turn) advance();
    // Reordered connections may admit the GEMM later; wait for its actual first
    // issue, still inside the same deadline and before allowing completion.
    if (client_bound) {
        while (outputs.empty()) advance();
    }
    const bool selected_expected =
        is_isolated    ? outputs.empty()
        : is_high_gemm ? outputs == std::vector<CUdeviceptr>{children[0]->output_address} &&
                             driver.physical_workers == std::vector<unsigned int>{320}
                       : outputs == std::vector<CUdeviceptr>{children[1]->output_address,
                                                             children[2]->output_address} &&
                             driver.physical_workers == std::vector<unsigned int>{160, 160};
    const bool bounded_metadata = global_metadata_queries <= count + 2;
    driver.on_launch = {};
    driver.query_status = CUDA_SUCCESS;
    for (std::size_t index = 0; index < count; ++index) send_byte(children[index]->socket, 'T');
    while (!service.finished()) advance();
    for (std::size_t index = 0; index < count; ++index) {
        wait_byte(children[index]->socket, 'H');
        children[index]->join();
    }
    expect(service.request_stop() && domain->snapshot().is_shutdown &&
               !domain->snapshot().is_quarantined && driver.live_resources() == 0 &&
               driver.context_creates == 1 && driver.context_destroys == 1 &&
               mixed_gemm.creates == 1 && mixed_gemm.destroys == 1 && !mixed_gemm.is_live &&
               mixed_gemm.lookups == 25 && !mixed_gemm.native_calls &&
               mixed_gemm.launches == (is_isolated ? 0U : 1U),
           "mixed paths complete full outputs and normal SDK/child/domain cleanup BEFORE policy "
           "assertions");
    if (audit) {
        const auto output = audit->text.str();
        const auto session = std::to_string(client_bound ? count : 1);
        if (client_bound) {
            const auto peer = *config.sgemm_audit_target->client;
            const auto binding = "SGEMM target binding: v=1, pid=" + std::to_string(peer.pid) +
                                 ", start=" + std::to_string(peer.start_time_ticks) +
                                 ", uid=" + std::to_string(peer.uid) + ", session=" + session;
            expect(output.find(binding) < output.find("SGEMM diagnostic: v=1, event=start"),
                   "authenticated training binding precedes the resolved diagnostic start");
        }
        expect(output.find("saturated=0, supported=0") != std::string::npos,
               "paired service audit cannot qualify as a complete issue range");
        const std::string expected =
            "SGEMM diagnostic: v=1, event=descriptor, session=" + session +
            ", sequence=1, transa=0, "
            "transb=0, m=1, n=1, k=1, lda=1, ldb=1, ldc=1, device_scalars=0, replacement=1";
        if (misses_target) {
            expect(
                service.audit_healthy() &&
                    output.find("SGEMM diagnostic: v=1, event=descriptor") == std::string::npos &&
                    output.find("SGEMM diagnostic: v=1, event=end, session=" + session +
                                ", sequence=4, captured=0") != std::string::npos,
                "a missed target is explicit and does not change the actual GEMM result");
            expect(
                !sgemm_range_pushes && !sgemm_range_pops &&
                    output.find("SGEMM range diagnostic: v=1, event=end, attempts=0, resumes=0") !=
                        std::string::npos,
                "a missing target cannot produce a profiler range");
        } else {
            expect(service.audit_healthy() && output.find(expected) != std::string::npos &&
                       output.find(expected, output.find(expected) + 1) == std::string::npos &&
                       output.find("SGEMM diagnostic: v=1, event=end, session=" + session +
                                   ", sequence=1, captured=1") != std::string::npos,
                   "one accepted SGEMM descriptor and healthy closure identify the exact target");
            expect(sgemm_range_pushes == 1 && sgemm_range_pops == 1 &&
                       output.find("SGEMM range diagnostic: v=1, event=end, attempts=1, resumes=0, "
                                   "push=0, pop=0, issued=1, native=0, completion=0") !=
                           std::string::npos,
                   "the unique real issue is wrapped exactly once on its owner");
        }
    }
    if (is_isolated) {
        expect(!service.save_pair_profile() &&
                   ::access(config.paired->profile_output.c_str(), F_OK) != 0 && errno == ENOENT,
               "a mixed isolated input cannot publish any successful pair-profile archive");
    } else {
        expect(::unlink(config.paired->profile_input.c_str()) == 0,
               "remove only this idle reproducible pair input");
    }
    expect(::unlink(config.profile_input.c_str()) == 0,
           "remove only this idle reproducible single input");
    expect(selected_expected,
           "a genuine private GEMM must neither erase the trained public pair nor bypass "
           "priority/profile gates");
    expect(bounded_metadata, "mixed candidate catalog preparation retains the N+2 SDK-query bound");
}

void invalid_profile_startup(const ExecutionServiceConfig& valid) {
    for (unsigned int scenario = 0; scenario < 9; ++scenario) {
        auto config = valid;
        if (scenario == 0) config.paired->profile_output.clear();
        if (scenario == 1) config.client_priorities[1].priority = 1;
        if (scenario == 2)
            config.client_priorities[1].kernel_mode = ExecutionClientKernelMode::kOriginal;
        if (scenario == 3) config.client_priorities.pop_back();
        if (scenario == 4) config.paired->profile.max_candidates = 3;
        if (scenario == 5) config.paired->profile_output = config.profile_input;
        if (scenario == 6) config.paired->is_isolated_profile = false;
        if (scenario == 7) config.uses_nccl = true;
        if (scenario == 8) config.paired->should_rebalance_on_arrival = true;
        bool rejected{};
        try {
            ExecutionService service(std::move(config), pair_dispatch());
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && driver.context_creates == 1 && !driver.context_destroys &&
                   !driver.launches && !driver.module_load_calls && !driver.allocation_calls &&
                   valid.paired->domain->snapshot().status == CUDA_SUCCESS &&
                   !valid.paired->domain->snapshot().is_closed &&
                   !valid.paired->domain->snapshot().attached_members &&
                   ::access(valid.transport.socket_path.c_str(), F_OK) < 0 && errno == ENOENT,
               "invalid isolated profile rejects before SDK/socket/domain transfer");
    }
}

unsigned int startup_initializations{};
CUresult CUDAAPI startup_initialize(unsigned int flags) {
    ++startup_initializations;
    return initialize(flags);
}
CUresult CUDAAPI rejected_startup_create(CUcontext* output, unsigned int, CUdevice) {
    *output = nullptr;
    return CUDA_ERROR_OUT_OF_MEMORY;
}
void workload_aware_startup(const std::string& path) {
    driver = {};
    startup_initializations = 0;
    const auto peer = current_process_identity();
    expect(peer.has_value(), "actual startup test process identity");
    auto callbacks = configuration_domain_dispatch();
    callbacks.initialize = startup_initialize;
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        callbacks, CudaExecutionDomainConfig{1, 0, CU_CTX_MAP_HOST, 4, 512});
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 4;
    config.session_limits = limits(1000);
    config.persistent.workers = 320;
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kWorkloadAware;
    config.client_priorities = {{*peer, 0, ExecutionClientKernelMode::kConfigured},
                                {{peer->pid + 1, peer->uid, peer->start_time_ticks + 1},
                                 0,
                                 ExecutionClientKernelMode::kConfigured}};
    config.profile.workers = {320};
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.profile_input = path + ".single";
    config.paired = ExecutionServicePairConfig{domain, {}, {}, {160}, false, {}, true, true};
    write_measurements(config, false);
    for (unsigned int scenario = 0; scenario < 12; ++scenario) {
        auto invalid = config;
        if (scenario == 0) invalid.paired.reset();
        if (scenario == 1) invalid.paired->should_rebalance_on_arrival = false;
        if (scenario == 2) invalid.profile_input += ".missing";
        if (scenario == 3) invalid.paired->workers = {0};
        if (scenario == 4) invalid.paired->workers = {400};
        if (scenario == 5) invalid.paired->profile.max_candidates = 0;
        if (scenario == 6)
            invalid.client_priorities[1].kernel_mode = ExecutionClientKernelMode::kOriginal;
        if (scenario == 7) invalid.paired->profile_input = path + ".missing-pair";
        if (scenario == 8) invalid.paired->domain.reset();
        if (scenario == 9) invalid.paired->should_initialize_domain = false;
        if (scenario == 10) invalid.paired->is_isolated_profile = true;
        if (scenario == 11) invalid.device_ordinal = -1;
        bool rejected{};
        try {
            ExecutionService service(std::move(invalid), pair_dispatch());
        } catch (const std::exception&) {
            rejected = true;
        }
        expect(rejected && !startup_initializations && !driver.context_creates &&
                   !driver.module_load_calls && !driver.allocation_calls &&
                   !domain->snapshot().has_context,
               "all CPU validation/archive errors precede native domain initialization");
    }
    {
        ExecutionService service(config, pair_dispatch());
        expect(startup_initializations == 1 && driver.context_creates == 1 &&
                   domain->snapshot().has_context && service.start(),
               "valid official-style ownership initializes once after complete CPU checks");
        expect(service.request_stop() && domain->snapshot().is_shutdown &&
                   driver.context_destroys == 1 && !driver.current && !driver.live_resources(),
               "empty owned workload-aware service shuts down its exact native context");
    }
    auto failed = config;
    failed.transport.socket_path = path + "/missing-parent/service.sock";
    failed.paired->domain = std::make_shared<CudaExecutionDomainOwner>(
        callbacks, CudaExecutionDomainConfig{1, 0, CU_CTX_MAP_HOST, 4, 512});
    {
        ExecutionService service(failed, pair_dispatch());
        expect(!service.start(), "a missing listener parent is a clean transport failure");
    }
    expect(startup_initializations == 2 && driver.context_creates == 2 &&
               driver.context_destroys == 2 && failed.paired->domain->snapshot().is_shutdown &&
               !driver.live_resources(),
           "listener failure still retires the initialized owned domain");
    callbacks.context_create = rejected_startup_create;
    failed.paired->domain = std::make_shared<CudaExecutionDomainOwner>(
        callbacks, CudaExecutionDomainConfig{1, 0, CU_CTX_MAP_HOST, 4, 512});
    bool rejected{};
    try {
        ExecutionService service(failed, pair_dispatch());
    } catch (const std::runtime_error& error) {
        rejected =
            std::string_view{error.what()}.find(
                "cannot initialize workload-aware execution domain: 2") != std::string_view::npos;
    }
    expect(rejected && startup_initializations == 3 && driver.context_creates == 2 &&
               driver.context_destroys == 2 && !failed.paired->domain->snapshot().has_context &&
               !failed.paired->domain->snapshot().is_quarantined && !driver.live_resources(),
           "known empty native creation error is exact, not a fabricated ready service or leak");
    expect(::unlink(config.profile_input.c_str()) == 0,
           "remove only exact idle startup measurement input");
}

void isolated_next_partner(const std::string& path, bool cancels_tail = false) {
    driver = {};
    const auto server = current_process_identity();
    expect(server.has_value(), "actual delayed partner service identity");
    Child left(path, *server, ExecutionContextPurpose::kRuntimePrimary, 2, false, false, false,
               CUDA_SUCCESS, 0, {}, false, true);
    Child right(path, *server, ExecutionContextPurpose::kRuntimePrimary, 2, false, false, false,
                CUDA_SUCCESS, 0, {}, cancels_tail, true);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        configuration_domain_dispatch(), CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 2, 512});
    expect(domain->initialize() == CUDA_SUCCESS, "delayed partner trusted domain");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 2;
    config.transport.max_pending_requests = 8;
    config.session_limits = limits(1000);
    config.persistent.workers = 320;
    config.client_priorities = {{left.identity(), 0, ExecutionClientKernelMode::kConfigured},
                                {right.identity(), 0, ExecutionClientKernelMode::kConfigured}};
    config.profile.workers = {320};
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.profile_input = path + ".single";
    config.paired = ExecutionServicePairConfig{
        domain, {.max_candidates = 1, .warmup_calls = 0, .samples_per_candidate = 1},
        {},     {160},
        true,   path + ".measured"};
    config.exit_when_idle = true;
    write_measurements(config, false);
    ExecutionService service(config, pair_dispatch());
    expect(service.start(), "start actual isolated delayed partner input");
    send_byte(left.socket, 'C');
    send_byte(right.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    const auto advance = [&] {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded delayed partner owner progress");
    };
    while (!left.ready || !right.ready) {
        advance();
        left.observe_ready();
        right.observe_ready();
    }
    driver.next_claimed = 8;
    send_byte(left.socket, 'G');
    send_byte(right.socket, 'G');
    wait_byte(left.socket, 'W');
    wait_byte(right.socket, 'W');
    while (service.pair_profile_snapshot().trained_pairs != 1) advance();
    expect(driver.launches == 2 && !service.pair_profile_snapshot().active,
           "first exact pair is fully measured and actually retired");
    if (cancels_tail) driver.query_status = CUDA_ERROR_NOT_READY;
    send_byte(left.socket, 'N');
    wait_byte(left.socket, 'P');
    for (unsigned int turn = 0; turn < 20; ++turn) advance();
    const bool waited_for_partner = driver.launches == 2 && driver.counter_initializations == 2;
    send_byte(right.socket, 'N');
    wait_byte(right.socket, 'P');
    bool retained_serial_tail = true;
    if (cancels_tail) {
        while (driver.launches < 3) advance();
        send_byte(right.socket, 'X');
        right.join();
        for (unsigned int turn = 0; turn < 20; ++turn) advance();
        retained_serial_tail = driver.launches == 3 && !domain->snapshot().is_shutdown;
        driver.query_status = CUDA_SUCCESS;
    }
    send_byte(left.socket, 'T');
    if (!cancels_tail) send_byte(right.socket, 'T');
    while (!service.finished()) advance();
    wait_byte(left.socket, 'H');
    if (!cancels_tail) wait_byte(right.socket, 'H');
    left.join();
    if (!cancels_tail) right.join();
    expect(service.request_stop() && service.save_pair_profile() &&
               domain->snapshot().is_shutdown && !domain->snapshot().is_quarantined &&
               driver.live_resources() == 0 && driver.context_creates == 1 &&
               driver.context_destroys == 1,
           "all old/new delayed-pair commands keep full outputs and normal owned cleanup");
    expect(::unlink(config.profile_input.c_str()) == 0 &&
               ::unlink(config.paired->profile_output.c_str()) == 0,
           "remove only exact owned idle reproducible profile outputs");
    expect(waited_for_partner,
           "trained isolated warmup allowed a later unpaired persistent command to issue");
    expect(retained_serial_tail && driver.launches == 4,
           "selected serial tail survives disconnect until actual first stop and exact issue");
}

void isolated_service_profiles(const std::string& path, bool has_partial, unsigned int commands,
                               bool disconnects = false, bool asymmetric_retirement = false,
                               CUresult expected_error = CUDA_SUCCESS) {
    driver = {};
    const auto server = current_process_identity();
    expect(server.has_value(), "isolated profile actual service identity");
    Child left(path, *server, ExecutionContextPurpose::kRuntimePrimary, commands, false, false,
               disconnects, expected_error),
        right(path, *server, ExecutionContextPurpose::kRuntimePrimary, commands, false, false,
              false, expected_error);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        configuration_domain_dispatch(), CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 2, 512});
    expect(domain->initialize() == CUDA_SUCCESS, "isolated service native domain");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 2;
    config.transport.max_pending_requests = 8;
    config.transport.io_timeout_ms = 10000;
    config.session_limits = limits(1000);
    config.persistent.workers = 320;
    config.client_priorities = {{left.identity(), 0, ExecutionClientKernelMode::kConfigured},
                                {right.identity(), 0, ExecutionClientKernelMode::kConfigured}};
    config.profile.workers = {320};
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.profile_input = path + ".single";
    config.paired = ExecutionServicePairConfig{
        domain, {.max_candidates = 4, .warmup_calls = 0, .samples_per_candidate = 1},
        {},     {80, 160},
        true,   path + ".measured"};
    config.exit_when_idle = true;
    if (expected_error == CUDA_ERROR_OUT_OF_MEMORY) config.paired->profile.max_bytes = 1;
    if (expected_error == CUDA_ERROR_NOT_SUPPORTED) {
        CudaProfileCache empty(config.profile, pair_clock, false);
        const auto identity = profile_runtime_identity();
        expect(identity.has_value(), "exact empty-original archive identity");
        const auto archive = empty.encode(*identity);
        expect(archive && write_profile_archive(config.profile_input, *archive),
               "valid empty originals are not qualified measurements for this input");
    } else {
        write_measurements(config, false);
    }
    invalid_profile_startup(config);
    ExecutionService service(config, pair_dispatch());
    expect(service.start() && !service.save_pair_profile(),
           "explicit isolated service cannot publish before shutdown");
    send_byte(left.socket, 'C');
    send_byte(right.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    while (!left.ready || !right.ready) {
        expect(service.step(1), "isolated clients prepare original resources");
        left.observe_ready();
        right.observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded isolated setup");
    }
    driver.defers_pair_metadata = asymmetric_retirement;
    send_byte(left.socket, 'G');
    send_byte(right.socket, 'G');
    wait_byte(left.socket, 'P');
    wait_byte(right.socket, 'P');
    driver.query_status = CUDA_ERROR_NOT_READY;
    driver.next_claimed = has_partial ? 3 : 8;
    if (expected_error != CUDA_SUCCESS) {
        while (service.pair_profile_status() == CUDA_SUCCESS) {
            expect(service.step(1), "return known profiling failure before any native issue");
            expect(std::chrono::steady_clock::now() < deadline,
                   "bounded actionable profile failure");
        }
        expect(service.pair_profile_status() == expected_error && !driver.launches &&
                   !service.pair_profile_snapshot().active && !service.save_pair_profile(),
               "permanent sampling failure is latched without native computation or false archive");
        driver.query_status = CUDA_SUCCESS;
        send_byte(left.socket, 'T');
        send_byte(right.socket, 'T');
        while (!service.finished()) {
            expect(service.step(1), "deliver each non-native failure and clean healthy owners");
            expect(std::chrono::steady_clock::now() < deadline, "bounded failed-profile cleanup");
        }
        wait_byte(left.socket, 'H');
        wait_byte(right.socket, 'H');
        left.join();
        right.join();
        expect(service.request_stop() && !service.save_pair_profile() &&
                   domain->snapshot().is_shutdown && driver.context_creates == 1 &&
                   driver.context_destroys == 1 && !driver.launches &&
                   !driver.counter_initializations && driver.live_resources() == 0 &&
                   !driver.current && ::access(config.paired->profile_output.c_str(), F_OK) < 0 &&
                   errno == ENOENT && ::unlink(config.profile_input.c_str()) == 0,
               "known pre-issue rejection does not quarantine, launch, leak or publish success");
        return;
    }
    if (asymmetric_retirement) {
        while (service.pending_priority_requests(0) < std::size_t{commands} * 2) {
            expect(service.step(1), "admit all posted inputs before permitting sample metadata");
            expect(std::chrono::steady_clock::now() < deadline, "bounded pre-sample admission");
        }
        expect(!driver.launches && !service.pair_profile_snapshot().active,
               "metadata deferral does not issue a default or invent a sample");
        driver.defers_pair_metadata = false;
    }
    while (driver.launches < 2) {
        expect(service.step(1), "reserve and issue only the first exact isolated pair");
        expect(std::chrono::steady_clock::now() < deadline, "bounded isolated first pair");
    }
    for (unsigned int step = 0; step < 20; ++step)
        expect(service.step(1), "hold actual profile tails while later input is already queued");
    const auto held = service.pair_profile_snapshot();
    expect(driver.launches == 2 && held.is_valid && held.is_learning && held.active == 1 &&
               !held.samples && driver.physical_workers == std::vector<unsigned int>{80, 80} &&
               driver.counter_initializations == 2 && !service.save_pair_profile(),
           "host replies and later queued commands cannot close, repeat or contaminate a sample");
    if (asymmetric_retirement) {
        driver.ready_launch = 1;
        for (unsigned int step = 0; step < 20; ++step)
            expect(service.step(1),
                   "retire one selected logical leg while its partner stays physical");
        const auto half = service.pair_profile_snapshot();
        expect(
            driver.launches == 2 && half.active == 1 && !half.samples && !half.rejected_samples &&
                !driver.context_destroys,
            "retired selected command is idle, not an invalid handle or next-command permission");
        driver.ready_launch = 0;
    }
    if (disconnects) {
        wait_byte(left.socket, 'H');
        send_byte(left.socket, 'X');
        left.join();
        for (unsigned int step = 0; step < 20; ++step)
            expect(service.step(1), "observe disconnect without inventing physical retirement");
        const auto abandoned = service.pair_profile_snapshot();
        expect(!abandoned.active && !abandoned.samples && abandoned.rejected_samples == 1 &&
                   driver.launches == 2 && !driver.context_destroys && !driver.host_frees,
               "disconnect invalidates measurement but retains both actual physical owners");
    }
    driver.next_claimed = 8;
    driver.query_status = CUDA_SUCCESS;
    if (!disconnects) send_byte(left.socket, 'T');
    send_byte(right.socket, 'T');
    while (!service.finished()) {
        expect(service.step(1), "complete unique isolated inputs and original full readbacks");
        expect(std::chrono::steady_clock::now() < deadline, "bounded isolated drain");
    }
    if (!disconnects) wait_byte(left.socket, 'H');
    wait_byte(right.socket, 'H');
    if (!disconnects) left.join();
    right.join();
    const auto measured = service.pair_profile_snapshot();
    expect(service.request_stop() && domain->snapshot().is_shutdown &&
               driver.context_creates == 1 && driver.context_destroys == 1 &&
               driver.live_resources() == 0 && !driver.current &&
               measured.samples == (disconnects ? 0 : commands) && !measured.active &&
               measured.rejected_samples == (disconnects ? 1U : 0U) && !measured.cancelled &&
               driver.counter_initializations == commands * 2 &&
               driver.launches == commands * 2 + (has_partial ? 2 : 0),
           "one observation per logical command pair; same counters survive partial epochs");
    const bool is_complete = commands == 4 && !disconnects;
    expect(service.save_pair_profile() == is_complete,
           "only a complete trained catalog may publish its protected archive");
    if (is_complete) {
        const auto identity = profile_runtime_identity();
        const auto archive =
            read_execution_archive(config.paired->profile_output, kCudaPairProfileArchiveMaxBytes);
        CudaPairProfileCache restored(pair_singles(), config.paired->profile, nullptr, false);
        expect(identity && archive && restored.restore(*archive, *identity) &&
                   restored.snapshot().loaded_pairs == 1 &&
                   restored.snapshot().trained_pairs == 1 && !service.save_pair_profile(),
               "actual-observer archive restores readonly and cannot overwrite an existing file");
        const std::vector<unsigned int> expected =
            has_partial ? std::vector<unsigned int>{80, 80, 80, 80, 80, 160, 160, 80, 160, 160}
                        : std::vector<unsigned int>{80, 80, 80, 160, 160, 80, 160, 160};
        expect(driver.physical_workers == expected,
               "actual resource candidates execute in order; partial work retains its geometry");
        expect(::unlink(config.paired->profile_output.c_str()) == 0,
               "remove only the known idle temporary profile output");
    } else {
        expect(::access(config.paired->profile_output.c_str(), F_OK) < 0 && errno == ENOENT,
               "an incomplete catalog creates no success archive");
    }
    expect(::unlink(config.profile_input.c_str()) == 0,
           "remove only the known idle original measurement input");
}

void invalid_pair_startup(const ExecutionServiceConfig& valid) {
    for (unsigned int scenario = 0; scenario < 10; ++scenario) {
        auto config = valid;
        auto sdk = pair_dispatch();
        if (scenario == 0) config.paired->domain.reset();
        if (scenario == 1) config.paired->workers.clear();
        if (scenario == 2) config.paired->workers = {0};
        if (scenario == 3) config.paired->workers = {80, 80};
        if (scenario == 4) config.paired->workers = {400};
        if (scenario == 5) config.profile_input.clear();
        if (scenario == 6) config.paired->profile.max_candidates = 0;
        if (scenario == 7) sdk.occupancy = nullptr;
        if (scenario == 8) config.paired->profile_input += ".missing";
        if (scenario == 9) {
            config.paired->should_rebalance_on_arrival = true;
            config.client_priorities[1].kernel_mode = ExecutionClientKernelMode::kOriginal;
        }
        bool rejected{};
        try {
            ExecutionService service(std::move(config), sdk);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && driver.context_creates == 1 && !driver.context_destroys &&
                   !driver.launches && !driver.module_load_calls && !driver.allocation_calls &&
                   valid.paired->domain->snapshot().status == CUDA_SUCCESS &&
                   !valid.paired->domain->snapshot().is_closed &&
                   !valid.paired->domain->snapshot().attached_members &&
                   ::access(valid.transport.socket_path.c_str(), F_OK) < 0 && errno == ENOENT,
               "invalid trusted startup rejects before SDK/socket/domain ownership transfer");
    }
}

void service_pairs(const std::string& path, bool is_cold, ExecutionContextPurpose purpose,
                   bool has_high_arrival = false, bool changes_configuration = false,
                   bool independent_streams = false) {
    driver = {};
    const auto server = current_process_identity();
    expect(server.has_value(), "actual service identity");
    Child left(path, *server, purpose, independent_streams ? 2 : 1, false, independent_streams),
        right(path, *server, purpose);
    std::unique_ptr<Child> high;
    if (has_high_arrival) high = std::make_unique<Child>(path, *server, purpose, 2);
    std::unique_ptr<Child> modifier;
    if (changes_configuration) modifier = std::make_unique<Child>(path, *server, purpose, 0, true);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        configuration_domain_dispatch(), CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 4, 512});
    expect(domain->initialize() == CUDA_SUCCESS, "framework-owned cold native domain");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 4;
    config.transport.max_pending_requests = 4;
    config.transport.io_timeout_ms = 10000;
    config.session_limits = limits(1000);
    config.persistent.workers = 320;
    config.client_priorities = {{left.identity(), 0, ExecutionClientKernelMode::kConfigured},
                                {right.identity(), 0, ExecutionClientKernelMode::kConfigured}};
    if (high)
        config.client_priorities.push_back(
            {high->identity(), 100, ExecutionClientKernelMode::kAuto});
    if (modifier)
        config.client_priorities.push_back(
            {modifier->identity(), 0, ExecutionClientKernelMode::kOriginal});
    config.profile.workers = {320};
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.profile_input = path + ".single";
    config.paired = ExecutionServicePairConfig{
        domain,
        {.max_candidates = 4, .warmup_calls = 0, .samples_per_candidate = 1},
        path + ".pair",
        {80, 160}};
    config.exit_when_idle = true;
    write_measurements(config, is_cold);
    if (!is_cold && purpose == ExecutionContextPurpose::kRuntimePrimary && !high && !modifier)
        invalid_pair_startup(config);
    ExecutionService service(config, pair_dispatch());
    expect(service.start(), "actual authenticated paired service");
    send_byte(left.socket, 'C');
    send_byte(right.socket, 'C');
    if (high) send_byte(high->socket, 'C');
    if (modifier) send_byte(modifier->socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    while (!left.ready || !right.ready || (high && !high->ready) ||
           (modifier && !modifier->ready)) {
        expect(service.step(1), "bootstrap both logical owners");
        left.observe_ready();
        right.observe_ready();
        if (high) high->observe_ready();
        if (modifier) modifier->observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded setup");
    }
    const bool is_shared = purpose == ExecutionContextPurpose::kRuntimePrimary;
    expect(
        driver.context_creates == (is_shared ? 1U : 3U) &&
            domain->snapshot().attached_members == (is_shared ? (high || modifier) ? 3U : 2U : 0U),
        "only trusted Runtime-primary attaches; Driver/legacy/user contexts stay isolated");
    send_byte(left.socket, 'G');
    send_byte(right.socket, 'G');
    wait_byte(left.socket, 'P');
    wait_byte(right.socket, 'P');
    driver.query_status = CUDA_ERROR_NOT_READY;
    unsigned int right_launch{};
    if (independent_streams)
        driver.on_launch = [&] {
            driver.next_claimed = driver.launch_output == left.output_address ? 3 : 8;
            if (driver.launch_output == right.output_address) right_launch = driver.launches + 1;
        };
    if (is_cold || !is_shared) driver.next_claimed = 8;
    while (!driver.launches) {
        expect(service.step(1), "admit exact commands before paired scheduling");
        expect(std::chrono::steady_clock::now() < deadline, "bounded first issue");
    }
    const auto expected = is_shared && is_cold ? 1U : 2U;
    for (unsigned int step = 0; step < 20; ++step)
        expect(service.step(1), "hold actual physical tails");
    driver.on_launch = {};
    expect(
        driver.launches == expected && !driver.context_destroys && !driver.host_frees,
        "cold serial choice cannot leak second issue; no host success supplies physical cleanup");
    if (is_shared && !is_cold) {
        expect(driver.physical_workers == std::vector<unsigned int>{160, 160} &&
                   driver.previous_counters == std::vector<std::uint64_t>{0, 0},
               "actual service consumes both measured exact pair grants, not ordinary defaults");
    }
    if (independent_streams) {
        expect(right_launch != 0, "independent partner actual physical epoch");
        driver.ready_launch = right_launch;
        for (unsigned int step = 0; step < 20; ++step)
            expect(service.step(1),
                   "retire only the partner, not this Session's held private tail");
        expect(driver.launches == 2 && driver.previous_counters == std::vector<std::uint64_t>{0, 0},
               "a ready independent stream cannot overlap its own Session's private tail without a "
               "checked pair grant");
        driver.ready_launch = 0;
    }
    if (changes_configuration) {
        send_byte(modifier->socket, 'G');
        wait_byte(modifier->socket, 'P');
        wait_byte(modifier->socket, 'H');
        send_byte(modifier->socket, 'M');
        wait_byte(modifier->socket, 'Q');
        while (!service.pending_priority_requests(0)) {
            expect(service.step(1),
                   "observe posted configuration demand before releasing low tails");
            expect(std::chrono::steady_clock::now() < deadline,
                   "bounded posted configuration intake");
        }
        for (unsigned int step = 0; step < 10; ++step)
            expect(service.step(1), "configuration waits for actual cohort quiescence");
        expect(!driver.context_control_calls && driver.launches == 2,
               "no requested setter or resume while actual pair tails are held");
        driver.on_launch = [] {
            if (driver.launches == 2) driver.query_status = CUDA_ERROR_NOT_READY;
        };
        driver.query_status = CUDA_SUCCESS;
        driver.next_claimed = 8;
        while (driver.launches < 3) {
            expect(service.step(1),
                   "change one logical configuration and resume exact retained work");
            expect(std::chrono::steady_clock::now() < deadline, "bounded configuration transition");
        }
        driver.on_launch = {};
        for (unsigned int step = 0; step < 10; ++step)
            expect(service.step(1),
                   "defer inactive sibling without quarantine or synthetic SDK failure");
        expect(driver.launches == 3 && driver.context_control_calls > 0,
               "different sibling configuration cannot issue beside retained physical work");
        wait_byte(modifier->socket, 'U');
        send_byte(modifier->socket, 'T');
        for (unsigned int step = 0; step < 10; ++step)
            expect(service.step(1),
                   "hold deferred logical query while another configuration is physical");
        expect(driver.launches == 3,
               "query deferral does not invent completion or restart either low command");
    }
    if (high) {
        send_byte(high->socket, 'G');
        wait_byte(high->socket, 'P');
        while (!service.pending_priority_requests(100)) {
            expect(service.step(1), "receive authenticated late high kernel demand");
            expect(std::chrono::steady_clock::now() < deadline, "bounded high demand");
        }
        for (unsigned int step = 0; step < 5; ++step)
            expect(service.step(1), "retreat cannot supply low stop");
        expect(driver.launches == 2, "high waits for BOTH actual low physical stops");
        driver.query_status = CUDA_SUCCESS;
        driver.on_launch = [] {
            if (driver.launches == 2) driver.query_status = CUDA_ERROR_NOT_READY;
        };
        while (driver.launches < 4) {
            expect(service.step(1), "pipeline high original kernels under exact fresh grants");
            expect(std::chrono::steady_clock::now() < deadline, "bounded high pipeline");
        }
        driver.on_launch = {};
        for (unsigned int step = 0; step < 10; ++step)
            expect(service.step(1), "hold actual high physical tails");
        expect(driver.launches == 4 &&
                   driver.previous_counters == std::vector<std::uint64_t>{0, 0} &&
                   driver.physical_workers == std::vector<unsigned int>{160, 160, 8, 8},
               "two native high calls pipeline before completion; low pair cannot resume early");
        send_byte(high->socket, 'T');
    }
    driver.next_claimed = 8;
    driver.query_status = CUDA_SUCCESS;
    send_byte(left.socket, 'T');
    send_byte(right.socket, 'T');
    while (!service.finished()) {
        expect(service.step(1), "retained once-only completion and cleanup");
        expect(std::chrono::steady_clock::now() < deadline, "bounded exact service drain");
    }
    // H may have arrived before T; each child owns its independent response path.
    wait_byte(left.socket, 'H');
    wait_byte(right.socket, 'H');
    left.join();
    right.join();
    if (high) {
        wait_byte(high->socket, 'H');
        high->join();
    }
    if (modifier) modifier->join();
    expect(service.request_stop() && domain->snapshot().is_shutdown &&
               driver.context_destroys == driver.context_creates && driver.live_resources() == 0 &&
               !driver.current,
           "member cleanup and last physical domain destruction are explicit and once-only");
    if (is_shared && !is_cold)
        expect(driver.previous_counters == (independent_streams
                                                ? std::vector<std::uint64_t>{0, 0, 3, 0}
                                                : std::vector<std::uint64_t>{0, 0, 3, 3}),
               "same independent counters survive paired partial continuation");
    if (is_shared && is_cold)
        expect(driver.previous_counters.empty() && driver.counter_initializations == 0 &&
                   driver.launches == 2,
               "cold serial originals never initialize or replay private computation");
    expect(::unlink(config.profile_input.c_str()) == 0 &&
               ::unlink(config.paired->profile_input.c_str()) == 0,
           "remove only known idle temporary measurement inputs");
}
void pointer_selection_lifecycle(const std::string& path, bool skips_driver_query) {
    driver = {};
    const auto server = current_process_identity();
    expect(server.has_value(), "actual metadata lifecycle service identity");
    Child left(path, *server, ExecutionContextPurpose::kRuntimePrimary, 0, false, false, false,
               CUDA_SUCCESS, skips_driver_query ? 2 : 1),
        right(path, *server, ExecutionContextPurpose::kRuntimePrimary);
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        configuration_domain_dispatch(), CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 4, 512});
    expect(domain->initialize() == CUDA_SUCCESS, "fresh shared physical owner");
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 8;
    config.transport.max_pending_requests = 8;
    config.transport.io_timeout_ms = 10000;
    config.session_limits = limits(1000);
    config.persistent.workers = 320;
    config.client_priorities = {{left.identity(), 0, ExecutionClientKernelMode::kConfigured},
                                {right.identity(), 0, ExecutionClientKernelMode::kConfigured}};
    config.profile.workers = {320};
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.profile_input = path + ".single";
    config.paired = ExecutionServicePairConfig{
        domain,
        {.max_candidates = 4, .warmup_calls = 0, .samples_per_candidate = 1},
        {},
        {80, 160}};
    config.exit_when_idle = true;
    write_measurements(config, false);
    auto callbacks = pair_dispatch();
    callbacks.pointer_get_attribute = [](void* output, CUpointer_attribute attribute, CUdeviceptr) {
        expect(attribute == CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL && driver.current,
               "actual original ordinal getter under selected query owner");
        *static_cast<int*>(output) = 0;
        return CUDA_SUCCESS;
    };
    callbacks.runtime.version = [](int* output) {
        *output = 12000;  // Explicit preparation rejection AFTER native selection.
        return cudaSuccess;
    };
    callbacks.runtime.pointer_attributes = [](cudaPointerAttributes*, const void*) {
        throw std::runtime_error("version rejection must precede requested Runtime getter");
        return cudaErrorUnknown;
    };
    ExecutionService service(config, callbacks);
    expect(service.start(), "actual metadata/common-domain service starts");
    send_byte(left.socket, 'C');
    send_byte(right.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{20};
    while (!left.ready || !right.ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded shared/isolated factory and pointer selection setup");
        left.observe_ready();
        right.observe_ready();
    }
    expect(driver.context_creates == 4 && driver.context_destroys == 3 &&
               domain->snapshot().attached_members == 2,
           "three isolated user contexts retire without destroying either shared member");
    driver.requires_explicit_cleanup = true;
    driver.ready_launch = 0;
    driver.query_status = CUDA_ERROR_NOT_READY;
    driver.next_claimed = 3;
    send_byte(right.socket, 'G');
    bool has_started{};
    while (!has_started) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "actual sibling original command starts");
        has_started = driver.launches != 0;
    }
    const auto cleanup_calls = driver.cleanup_calls();
    const auto control_calls = driver.context_control_calls;
    send_byte(left.socket, 'G');
    bool has_close{};
    while (!has_close) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "receive actual pending Close");
        char value{};
        const auto received = ::recv(left.socket, &value, 1, MSG_DONTWAIT);
        if (received == 1) {
            expect(value == 'Q', "exact primary retirement request phase");
            has_close = true;
        } else {
            expect(received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR),
                   "owned client remains live while requesting retirement");
        }
    }
    for (unsigned int turn = 0; turn < 10; ++turn)
        expect(service.step(0), "held physical tail observation");
    expect(driver.cleanup_calls() == cleanup_calls &&
               driver.context_control_calls == control_calls && driver.context_destroys == 3 &&
               domain->snapshot().attached_members == 2 && driver.counter_initializations == 1,
           "Close/configuration cannot release or mutate a sibling's held physical reservation");
    driver.next_claimed = 8;
    driver.ready_launch = 100;
    driver.query_status = CUDA_SUCCESS;
    bool has_rejoined{}, has_finished{};
    while (!has_finished) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded member-only close, fresh generation rejoin and exact execution");
        char value{};
        const auto received = ::recv(left.socket, &value, 1, MSG_DONTWAIT);
        if (received == 1) {
            if (value == 'U')
                has_rejoined = true;
            else {
                expect(value == 'D' && has_rejoined, "fresh generation numerical cleanup phase");
                has_finished = true;
            }
        } else {
            expect(received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR),
                   "owned lifecycle client remains live until full checks finish");
        }
    }
    left.join();
    expect(domain->snapshot().attached_members == 1 && driver.context_creates == 4 &&
               driver.context_destroys == 3 && driver.counter_initializations == 2,
           "old/fresh member cleanup preserves the sibling and one shared native context");
    send_byte(right.socket, 'T');
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "final sibling full output and domain cleanup");
    right.join();
    expect(domain->snapshot().is_shutdown && !domain->snapshot().is_quarantined &&
               driver.context_creates == 4 && driver.context_destroys == 4 &&
               driver.live_resources() == 0 && driver.launches == 3 &&
               driver.physical_workers == std::vector<unsigned int>{320, 320, 320} &&
               driver.previous_counters == std::vector<std::uint64_t>{0, 3, 0},
           "exact sibling continuation and fresh independent counter survive primary retirement");
    expect(::unlink(config.profile_input.c_str()) == 0,
           "remove only idle owned reproducible measurement input");
}
namespace priority_space_share {
enum class Scenario { kPreparedOrder, kAsyncDrain, kDeparture, kLateArrival };
struct NativeStream {
    CUstream stream{};
    int priority{};
};
std::vector<NativeStream> streams;
std::vector<CUdeviceptr> outputs;
unsigned int initialization_calls{};
CUresult CUDAAPI tracked_initialize(unsigned int flags) {
    ++initialization_calls;
    return initialize(flags);
}
// Required original SDK callback signatures.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CUresult CUDAAPI range(int* least, int* greatest) {
    expect(driver.current, "range query uses the actual selected common context");
    *least = 0;
    *greatest = -3;
    return CUDA_SUCCESS;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CUresult CUDAAPI create(CUstream* output, unsigned int flags, int priority) {
    const auto status = stream_create(output, flags);
    if (status == CUDA_SUCCESS) streams.push_back({*output, std::clamp(priority, -3, 0)});
    return status;
}
CUresult CUDAAPI query(CUstream stream, int* output) {
    const auto found = std::ranges::find(streams, stream, &NativeStream::stream);
    expect(found != streams.end(), "query an actual source-owned native stream");
    *output = found->priority;
    return CUDA_SUCCESS;
}
void rejects_invalid_startup(const ExecutionServiceConfig& valid,
                             const CudaSessionDispatch& callbacks) {
    for (unsigned int mode = 0; mode < 14; ++mode) {
        auto config = valid;
        auto native = callbacks;
        if (mode == 0) config.shared.reset();
        if (mode == 1) config.shared->domain.reset();
        if (mode == 2) config.uses_original_kernels = false;
        if (mode == 3) config.persistent.workers = 1;
        if (mode == 4) config.client_priorities.resize(1);
        if (mode == 5) config.shared->should_initialize_domain = false;
        if (mode == 6) config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
        if (mode == 7) config.paired.emplace();
        if (mode == 8) native.context.priority_range = nullptr;
        if (mode == 9) native.stream_create_priority = nullptr;
        if (mode == 10) native.stream_get_priority = nullptr;
        if (mode == 11) native.stream_wait_event = nullptr;
        if (mode == 12) config.persistent.replace_sgemm = true;
        if (mode == 13) config.profile_output = "/nonexistent/space-share-profile";
        bool rejected{};
        try {
            ExecutionService invalid(config, native);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && !initialization_calls && !driver.context_creates &&
                   !valid.shared->domain->snapshot().has_context,
               "invalid sharing assembly rejects before native initialization or domain mutation");
    }
}
void run(const std::string& path, Scenario scenario = Scenario::kPreparedOrder,
         bool uses_mixed_contexts = false, unsigned int initial_flags = 0) {
    driver = {};
    streams.clear();
    outputs.clear();
    initialization_calls = 0;
    const auto server = current_process_identity();
    expect(server.has_value(), "exact service process identity");
    const bool asynchronous = scenario == Scenario::kAsyncDrain;
    const bool departure = scenario == Scenario::kDeparture;
    const bool arrival = scenario == Scenario::kLateArrival;
    const auto low_commands = departure ? 3U : asynchronous || arrival ? 1U : 2U;
    const auto first_commands = departure ? 0U : asynchronous ? 2U : 1U;
    const auto second_commands = departure ? 0U : asynchronous ? 2U : 1U;
    const ClientContextSettings context_settings{initial_flags, uses_mixed_contexts};
    const auto low_purpose = uses_mixed_contexts ? ExecutionContextPurpose::kProcessPrimary
                                                 : ExecutionContextPurpose::kRuntimePrimary;
    const auto high_purpose = uses_mixed_contexts ? ExecutionContextPurpose::kExplicitUser
                                                  : ExecutionContextPurpose::kRuntimePrimary;
    Child low(path, *server, low_purpose, low_commands, false, false, false, CUDA_SUCCESS, 0, {},
              false, false, false, !asynchronous && !arrival, asynchronous, false,
              context_settings);
    Child first(path, *server, high_purpose, first_commands, false, false, false, CUDA_SUCCESS, 0,
                {}, false, false, false, false, asynchronous, asynchronous, context_settings);
    Child second(path, *server, ExecutionContextPurpose::kRuntimePrimary, second_commands, false,
                 false, false, CUDA_SUCCESS, 0, {}, false, false, false, asynchronous, asynchronous,
                 false, context_settings);
    auto domain_callbacks = configuration_domain_dispatch();
    domain_callbacks.initialize = tracked_initialize;
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        domain_callbacks, CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 4, 512});
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 8;
    config.transport.max_pending_requests = 8;
    config.transport.io_timeout_ms = 10000;
    config.session_limits = limits(1000);
    config.session_limits.submissions.max_commands = 8;
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kPrioritySpaceShare;
    config.uses_original_kernels = true;
    config.client_priorities = {{low.identity(), 0},
                                {first.identity(), 100},
                                {second.identity(), asynchronous ? 0U : 100U}};
    config.shared = ExecutionServiceSharedConfig{domain, true};
    config.exit_when_idle = true;
    auto callbacks = session_dispatch();
    callbacks.initialize = tracked_initialize;
    callbacks.context.priority_range = range;
    callbacks.stream_create_priority = create;
    callbacks.stream_get_priority = query;
    expect(!driver.context_creates, "cold assembly has no SDK context creation");
    rejects_invalid_startup(config, callbacks);
    bool diagnostic_reentry_rejected = true;
    ExecutionService service(config, callbacks);
    expect(!service.pending_priority_preparations(100) &&
               !service.pending_priority_async_admissions(100),
           "an unstarted owner cannot supply pending preparation observations");
    expect(service.start(), "actual original space-sharing service starts without profiles");
    // The independent SDK fixture validates the actual extra-buffer ABI first.
    // Reentry into this readonly diagnostic cannot inspect an active SDK turn.
    driver.on_launch = [&] {
        outputs.push_back(driver.launch_output);
        diagnostic_reentry_rejected = diagnostic_reentry_rejected &&
                                      !service.pending_priority_preparations(100) &&
                                      !service.pending_priority_async_admissions(100);
    };
    for (auto* child : {&low, &first, &second}) send_byte(child->socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    while (!low.ready || !first.ready || !second.ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded original resources and callback setup");
        low.observe_ready();
        first.observe_ready();
        second.observe_ready();
    }
    // One domain initialization plus each original client bootstrap's cuInit;
    // only the domain may create the physical native context.
    expect(initialization_calls == 4 && driver.context_creates == 1 &&
               domain->snapshot().attached_members == 3 && streams.size() == 3 &&
               std::ranges::count(streams, 0, &NativeStream::priority) == (asynchronous ? 2 : 1) &&
               std::ranges::count(streams, -3, &NativeStream::priority) == (asynchronous ? 1 : 2),
           "three original clients share one native owner with exact hidden stream priorities");
    const auto step = [&] {
        expect(service.step(0) && std::chrono::steady_clock::now() < deadline,
               "bounded actual original scheduling and physical-owner progress");
    };
    const auto receive_phase = [&](Child& child, char expected) {
        while (true) {
            char value{};
            const auto received = ::recv(child.socket, &value, 1, MSG_DONTWAIT);
            if (received == 1) {
                expect(value == expected, "exact child phase, never a fabricated execution grant");
                break;
            }
            expect(received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR),
                   "owned child remains alive during its bounded phase");
            step();
        }
    };
    const auto callback_permission = [&](Child& child) {
        send_byte(child.socket, 'F');
        receive_phase(child, 'Q');
    };
    bool policy_verified{};
    bool held_physical_verified{};
    if (asynchronous || arrival) {
        driver.query_status = CUDA_ERROR_NOT_READY;
        driver.ready_launch = 0;
        send_byte(low.socket, 'G');
        wait_byte(low.socket, 'P');
        while (outputs.empty()) step();
        expect(outputs.size() == 1 && outputs.front() == low.output_address,
               "actual initial low original precedes all high demand");
        const auto held_cache = driver.cache;
        const auto held_control_calls = driver.context_control_calls;
        for (auto* child : {&first, &second}) send_byte(child->socket, 'G');
        wait_byte(first.socket, 'P');
        wait_byte(second.socket, 'P');
        if (asynchronous) {
            callback_permission(second);
            driver.next_context_query_status = CUDA_ERROR_INVALID_CONTEXT;
            const auto pending = service.pending_priority_async_admissions(100);
            expect(pending == 2 && service.pending_priority_preparations(100) == 0 &&
                       outputs.size() == 1 && driver.cache == held_cache &&
                       driver.context_control_calls == held_control_calls &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT,
                   "two exact accepted originals need no preparation but remain GPU-unissued "
                   "behind the independently held incompatible lower configuration");
            expect(service.pending_priority_async_admissions(100) == pending &&
                       outputs.size() == 1 && driver.cache == held_cache &&
                       driver.context_control_calls == held_control_calls &&
                       driver.next_context_query_status == CUDA_ERROR_INVALID_CONTEXT,
                   "readonly exact admission count cannot query, materialize or issue native work");
            driver.next_context_query_status = CUDA_SUCCESS;
            driver.ready_launch = 1;
            send_byte(second.socket, 'E');
            std::size_t most_high_per_turn{};
            while (std::ranges::count(outputs, first.output_address) < 2) {
                const auto before = std::ranges::count(outputs, first.output_address);
                step();
                const auto added = std::ranges::count(outputs, first.output_address) - before;
                most_high_per_turn = std::max(most_high_per_turn, static_cast<std::size_t>(added));
            }
            policy_verified =
                most_high_per_turn == 2 && service.pending_priority_preparations(100) == 0 &&
                service.pending_priority_async_admissions(100) == 0 && outputs.size() == 3 &&
                outputs[1] == first.output_address && outputs[2] == first.output_address;
            held_physical_verified =
                std::ranges::count_if(driver.events, [](const auto& event) {
                    return event.is_live && event.is_recorded && event.launch > 1;
                }) >= 2;
            step();  // Harvest/ACK the exact high host replies with both GPU tails still held.
            policy_verified = policy_verified && outputs.size() == 3 &&
                              service.pending_priority_async_admissions(100) == 0 &&
                              service.pending_priority_preparations(100) == 0;
            driver.query_status = CUDA_SUCCESS;
            while (outputs.size() < 5) step();
            policy_verified = policy_verified && outputs[3] == second.output_address &&
                              outputs[4] == second.output_address;
        } else {
            while (outputs.size() < 3) step();
            policy_verified = outputs[0] == low.output_address &&
                              outputs[1] != low.output_address &&
                              outputs[2] != low.output_address && outputs[1] != outputs[2];
            held_physical_verified = std::ranges::count_if(driver.events, [](const auto& event) {
                                         return event.is_live && event.is_recorded && event.launch;
                                     }) >= 3;
        }
    } else {
        send_byte(low.socket, 'G');
        wait_byte(low.socket, 'P');
        while (service.pending_priority_requests(0) < low_commands) step();
        callback_permission(low);
        expect(outputs.empty(), "queued low originals cannot pass the unreturned callback");
        for (auto* child : {&first, &second}) send_byte(child->socket, 'G');
        wait_byte(first.socket, 'P');
        wait_byte(second.socket, 'P');
        if (departure) {
            receive_phase(first, 'H');
            receive_phase(second, 'H');
            for (auto* child : {&first, &second}) send_byte(child->socket, 'T');
            while (domain->snapshot().attached_members != 1) step();
            first.join();
            second.join();
            step();  // Observe their normal socket departures, not just SDK returns.
            expect(driver.context_creates == 1 && !driver.context_destroys && outputs.empty(),
                   "departed high members cannot destroy the surviving low physical owner");
        }
        driver.query_status = CUDA_ERROR_NOT_READY;
        driver.ready_launch = 0;
        send_byte(low.socket, 'E');
        std::size_t low_steps{}, most_low_per_turn{};
        const auto wanted = departure ? 3U : 4U;
        while (outputs.size() < wanted) {
            const auto before = std::ranges::count(outputs, low.output_address);
            step();
            const auto added = std::ranges::count(outputs, low.output_address) - before;
            most_low_per_turn = std::max(most_low_per_turn, static_cast<std::size_t>(added));
            if (added) ++low_steps;
        }
        policy_verified =
            departure
                ? most_low_per_turn == 3 && low_steps == 1
                : most_low_per_turn == 1 && low_steps == 2 && outputs[0] != low.output_address &&
                      outputs[1] != low.output_address && outputs[0] != outputs[1] &&
                      outputs[2] == low.output_address && outputs[3] == low.output_address;
        held_physical_verified = std::ranges::count_if(driver.events, [](const auto& event) {
                                     return event.is_live && event.is_recorded && event.launch;
                                 }) >= wanted;
    }
    const bool original_geometry_verified =
        !driver.counter_initializations &&
        std::ranges::all_of(driver.physical_workers, [](auto blocks) { return blocks == 8; });
    driver.requires_explicit_cleanup = true;
    driver.query_status = CUDA_SUCCESS;
    send_byte(low.socket, 'T');
    if (!departure)
        for (auto* child : {&first, &second}) send_byte(child->socket, 'T');
    while (!service.finished())
        expect(service.step(0) && std::chrono::steady_clock::now() < deadline,
               "complete full numerical readback and exact member/domain retirement");
    low.join();
    if (!departure) {
        first.join();
        second.join();
    }
    expect(domain->snapshot().is_shutdown && !driver.live_resources() &&
               driver.context_creates == 1 && driver.context_destroys == 1,
           "all original values pass before one common owner destroys after the last member");
    driver.on_launch = {};
    expect(diagnostic_reentry_rejected && service.pending_priority_preparations(100) == 0 &&
               service.pending_priority_async_admissions(100) == 0,
           "diagnostic rejects active native reentry and observes actual empty normal retirement");
    expect(original_geometry_verified && held_physical_verified,
           "original geometry and independently held physical events, without retreat/private "
           "counters");
    expect(
        policy_verified,
        "after complete numerical/resource cleanup: current highest drains all queued work; "
        "lower clients retain their bounded turn and late high originals do not wait for low stop");
}
}  // namespace priority_space_share
}  // namespace
int main(int argc, char** argv) {
    char pattern[] = "/tmp/glimmer-service-pair-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    try {
        const std::string root = directory;
        if (argc == 2) {
            const std::string_view mode = argv[1];
            if (mode == "--priority-space-share") {
                priority_space_share::run(root + "/space-share.sock");
                expect(::rmdir(directory) == 0,
                       "space-sharing fixture retires its owned directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--priority-space-share-mixed" ||
                mode == "--priority-space-share-mixed-flags" ||
                mode == "--priority-space-share-mixed-drain" ||
                mode == "--priority-space-share-mixed-departure" ||
                mode == "--priority-space-share-mixed-arrival") {
                const auto scenario = mode == "--priority-space-share-mixed-drain"
                                          ? priority_space_share::Scenario::kAsyncDrain
                                      : mode == "--priority-space-share-mixed-departure"
                                          ? priority_space_share::Scenario::kDeparture
                                      : mode == "--priority-space-share-mixed-arrival"
                                          ? priority_space_share::Scenario::kLateArrival
                                          : priority_space_share::Scenario::kPreparedOrder;
                priority_space_share::run(root + "/mixed.sock", scenario, true,
                                          mode == "--priority-space-share-mixed-flags"
                                              ? static_cast<unsigned int>(CU_CTX_SCHED_YIELD)
                                              : 0U);
                expect(::rmdir(directory) == 0,
                       "mixed context scenario retires its exact directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--priority-space-share-drain" ||
                mode == "--priority-space-share-departure" ||
                mode == "--priority-space-share-arrival") {
                const auto scenario = mode == "--priority-space-share-drain"
                                          ? priority_space_share::Scenario::kAsyncDrain
                                      : mode == "--priority-space-share-departure"
                                          ? priority_space_share::Scenario::kDeparture
                                          : priority_space_share::Scenario::kLateArrival;
                priority_space_share::run(root + "/space-share.sock", scenario);
                expect(::rmdir(directory) == 0, "new owner scenario retires its exact directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--workload-aware-startup") {
                workload_aware_startup(root + "/startup");
                expect(::rmdir(directory) == 0, "startup fixture cleans its owned directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--workload-aware-pairs") {
                global_service_pairs(root + "/aware-global", 0, true);
                global_service_pairs(root + "/aware-arrival", 5, true);
                expect(::rmdir(directory) == 0, "workload-aware cases clean their owned directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--isolated-next-partner") {
                isolated_next_partner(root + "/next-partner");
                isolated_next_partner(root + "/cancel-tail", true);
                expect(::rmdir(directory) == 0,
                       "delayed partner fixture cleans its owned directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--arrival-rebalance") {
                for (unsigned int scenario = 5; scenario <= 9; ++scenario)
                    global_service_pairs(root + "/arrival-" + std::to_string(scenario), scenario);
                expect(::rmdir(directory) == 0,
                       "arrival case cleans its owned temporary directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--global-pair-capacity") {
                global_service_pairs(root + "/capacity", 4);
                expect(::rmdir(directory) == 0, "bounded-cache case cleans its owned directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--global-pair-selection") {
                for (unsigned int scenario = 0; scenario < 4; ++scenario)
                    global_service_pairs(root + "/global-" + std::to_string(scenario), scenario);
                global_service_pairs(root + "/metadata-unsupported", 10);
                global_service_pairs(root + "/metadata-invalid", 11);
                expect(::rmdir(directory) == 0,
                       "global cases clean their owned temporary directory");
                return EXIT_SUCCESS;
            }
            if (mode == "--mixed-pair-candidates") {
                mixed_service_pairs(root + "/mixed", false);
                mixed_service_pairs(root + "/mixed-high", false, true);
                mixed_service_pairs(root + "/mixed-missing", false, true, true);
                mixed_service_pairs(root + "/mixed-client", false, true, false, true);
                mixed_service_pairs(root + "/mixed-client-missing", false, true, true, true);
                mixed_service_pairs(root + "/mixed-isolated", true);
                expect(::rmdir(directory) == 0,
                       "mixed-candidate fixtures remove their exact owned temporary directory");
                return EXIT_SUCCESS;
            }
            expect(
                mode == "--pointer-selection-driver" || mode == "--pointer-selection-preparation",
                "known explicit pointer selection test mode");
            pointer_selection_lifecycle(root + "/selection.sock",
                                        mode == "--pointer-selection-preparation");
            expect(::rmdir(directory) == 0, "retire only empty owned diagnostic directory");
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "no unexpected fixture arguments");
        pointer_selection_lifecycle(root + "/selection-driver.sock", false);
        pointer_selection_lifecycle(root + "/selection-preparation.sock", true);
        isolated_service_profiles(root + "/profile.sock", false, 4);
        isolated_service_profiles(root + "/profile-asymmetric.sock", false, 4, false, true);
        isolated_service_profiles(root + "/profile-partial.sock", true, 4);
        isolated_service_profiles(root + "/profile-incomplete.sock", false, 1);
        isolated_service_profiles(root + "/profile-disconnect.sock", true, 1, true);
        isolated_service_profiles(root + "/profile-capacity.sock", false, 3, false, false,
                                  CUDA_ERROR_OUT_OF_MEMORY);
        isolated_service_profiles(root + "/profile-missing.sock", false, 3, false, false,
                                  CUDA_ERROR_NOT_SUPPORTED);
        service_pairs(root + "/trained.sock", false, ExecutionContextPurpose::kRuntimePrimary);
        service_pairs(root + "/cold.sock", true, ExecutionContextPurpose::kRuntimePrimary);
        service_pairs(root + "/high.sock", false, ExecutionContextPurpose::kRuntimePrimary, true);
        service_pairs(root + "/streams.sock", false, ExecutionContextPurpose::kRuntimePrimary,
                      false, false, true);
        service_pairs(root + "/configuration.sock", false, ExecutionContextPurpose::kRuntimePrimary,
                      false, true);
        service_pairs(root + "/legacy.sock", false, ExecutionContextPurpose::kLegacyIsolated);
        service_pairs(root + "/driver.sock", false, ExecutionContextPurpose::kProcessPrimary);
        service_pairs(root + "/user.sock", false, ExecutionContextPurpose::kExplicitUser);
        expect(::rmdir(directory) == 0, "retire only the empty owned fixture directory");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Service pair composition: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
