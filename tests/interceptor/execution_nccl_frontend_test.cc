#include "glimmer/control/execution_nccl_protocol.h"
#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <nccl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <fcntl.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_nccl_frontend_native_calls();
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
constexpr unsigned int kHostileCases = 19;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "open independently named NCCL caller provider");
    constexpr std::array names{
        "ncclGetUniqueId",       "ncclCommInitRank", "ncclCommInitRankConfig",
        "ncclCommGetAsyncError", "ncclCommDestroy",  "ncclCommAbort",
        "ncclGroupStart",        "ncclGroupEnd",     "ncclAllGather",
        "ncclAllReduce",         "ncclBcast"};
    const std::array<void*, 11> direct{reinterpret_cast<void*>(&ncclGetUniqueId),
                                       reinterpret_cast<void*>(&ncclCommInitRank),
                                       reinterpret_cast<void*>(&ncclCommInitRankConfig),
                                       reinterpret_cast<void*>(&ncclCommGetAsyncError),
                                       reinterpret_cast<void*>(&ncclCommDestroy),
                                       reinterpret_cast<void*>(&ncclCommAbort),
                                       reinterpret_cast<void*>(&ncclGroupStart),
                                       reinterpret_cast<void*>(&ncclGroupEnd),
                                       reinterpret_cast<void*>(&ncclAllGather),
                                       reinterpret_cast<void*>(&ncclAllReduce),
                                       reinterpret_cast<void*>(&ncclBcast)};
    for (std::size_t index = 0; index != names.size(); ++index)
        expect(::dlsym(library, names[index]) == direct[index] &&
                   ::dlsym(RTLD_DEFAULT, names[index]) == direct[index],
               "all eleven global and explicit lookups select the controlled ABI");
    for (const auto name : {"ncclCommCount", "pncclAllGather", "ncclBroadcast"}) {
        (void)::dlerror();
        expect(!::dlsym(library, name) && ::dlerror(),
               "unsupported explicit opaque consumer rejects");
        expect(!::dlsym(RTLD_DEFAULT, name) && ::dlerror(),
               "unsupported global opaque consumer rejects");
    }
    expect(::dlclose(library) == 0, "caller provider can close without invalidating wrapper code");
}
void configure(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated socket endpoint before frontend construction");
}
ExecutionNcclObservation comm_write(ExecutionNcclCommunicator communicator) {
    ExecutionNcclObservation result;
    result.effects = 2;
    result.communicator = communicator;
    return result;
}
ExecutionNcclObservation uid_write(std::byte value) {
    ExecutionNcclObservation result;
    result.effects = 1;
    result.unique_id_mask.fill(std::byte{255});
    result.unique_id.fill(value);
    return result;
}
struct Mock {
    struct Import {
        std::optional<ExecutionSourceContinuation> origin;
        std::uint64_t helper{}, resource{}, next_cleanup{1};
        void* mapping{MAP_FAILED};
        std::size_t extent{};
        Import() = default;
        Import(const Import&) = delete;
        Import& operator=(const Import&) = delete;
        ~Import() {
            if (mapping != MAP_FAILED && ::munmap(mapping, extent) != 0) std::abort();
        }
    };
    std::unordered_map<std::uint64_t, std::uint64_t> sequences;
    std::unordered_map<std::uint64_t, std::uint64_t> ordinary_sequences;
    std::unordered_map<std::uint64_t, unsigned int> depths;
    std::unordered_map<std::uint64_t, std::uint64_t> groups;
    std::unordered_map<std::uint64_t, std::uint32_t> helper_roles;
    std::vector<std::unique_ptr<Import>> imports;
    std::array<std::uint32_t, 16> device_values{};
    std::vector<std::pair<ExecutionNcclScope, ExecutionNcclCommunicator>> pending;
    std::array<unsigned int, 11> operations{};
    std::uint64_t source{}, transport{}, next_operation{1}, next_comm{1}, next_stream{901};
    unsigned int nccl_hellos{}, source_hellos{}, owned_streams{}, defaults{}, legacies{};
    unsigned int ordinary_calls{}, clean_rejections{}, host_imports{}, staging_imports{},
        host_cleanups{}, staging_cleanups{}, transferred_values{};
    std::atomic<bool> rejects_next_allocate{};
    bool publishes_group_init_immediately{};
    ExecutionWireReply respond(ExecutionReceivedRequest& received) {
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        const auto effective = normalize_execution_request(received.message.header.operation,
                                                           received.message.payload);
        expect(effective.has_value(),
               "every request has a canonical one-level effective operation");
        const auto opcode = effective->operation;
        if (effective->continuation) {
            const auto& scope = *effective->continuation;
            expect(scope.caller[0] == static_cast<std::uint64_t>(::getpid()) && scope.group &&
                       opcode != ExecutionOpcode::kNcclCall,
                   "ordinary calls use the actual NCCL caller/group, never its public sequence");
            if (!scope.flags) {
                expect(depths[scope.caller[1]] && groups[scope.caller[1]] == scope.group &&
                           scope.ordinary_sequence == ++ordinary_sequences[scope.caller[1]],
                       "normal per-caller continuations arrive densely within their live group");
                ++ordinary_calls;
            }
        }
        ExecutionPayloadReader reader(effective->payload);
        if (opcode == ExecutionOpcode::kHello) {
            if (const auto descriptor = decode_execution_context_bootstrap(effective->payload)) {
                expect(descriptor->source_generation &&
                           descriptor->purpose != ExecutionContextPurpose::kLegacyIsolated,
                       "ordinary source preserves a typed generation and context purpose");
                if (!source) source = received.connection;
                ++source_hellos;
            } else {
                std::uint32_t first{}, flags{};
                expect(reader.read_u32(first), "read bootstrap tag");
                if (first == kExecutionNcclRole) {
                    expect(reader.read_u32(flags) && !flags && reader.empty(),
                           "NCL1 canonical role");
                    transport = received.connection;
                    ++nccl_hellos;
                } else if (first == kExecutionHostOwnerRole ||
                           first == kExecutionStagingOwnerRole) {
                    expect(reader.read_u32(flags) && reader.empty(),
                           "exact helper bootstrap flags");
                    if (effective->continuation)
                        expect(effective->continuation->origin_source == source,
                               "helper bootstrap retains the original real source, never its own "
                               "session");
                    helper_roles[received.connection] = first;
                    const auto ownership = encode_execution_bootstrap_ownership(
                        ExecutionBootstrapOwnership::kRetained);
                    expect(ownership.has_value(), "encode exact helper bootstrap ownership");
                    reply.data.assign(ownership->begin(), ownership->end());
                } else
                    expect(false, "unknown bootstrap is neither a helper nor ordinary context");
            }
            return reply;
        }
        if (opcode == ExecutionOpcode::kAllocate) {
            std::uint64_t bytes{};
            expect(reader.read_u64(bytes) && bytes == 64 && reader.empty(),
                   "exact source allocation");
            if (rejects_next_allocate.exchange(false)) {
                expect(effective->continuation.has_value(),
                       "reject only the intended normal wrapper");
                ++clean_rejections;
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = CUDA_ERROR_OUT_OF_MEMORY;
                return reply;
            }
            reply.id = 701;
            reply.address = 0x100000;
            return reply;
        }
        if (opcode == ExecutionOpcode::kImportHostMemory) {
            std::uint64_t bytes{}, extent{};
            std::uint32_t flags{};
            expect(reader.read_u64(bytes) && bytes == 64 && reader.read_u64(extent) &&
                       extent >= bytes && reader.read_u32(flags) && !flags && reader.empty() &&
                       received.descriptor.get() >= 0 && helper_roles.contains(received.connection),
                   "full ordinary import retains exact helper role, fields and real SCM_RIGHTS");
            auto imported = std::make_unique<Import>();
            imported->origin = effective->continuation;
            imported->helper = received.connection;
            imported->resource = 100 + imports.size();
            imported->extent = static_cast<std::size_t>(extent);
            imported->mapping = ::mmap(nullptr, imported->extent, PROT_READ | PROT_WRITE,
                                       MAP_SHARED, received.descriptor.get(), 0);
            expect(imported->mapping != MAP_FAILED, "actual descriptor-backed complete host bytes");
            if (helper_roles[received.connection] == kExecutionHostOwnerRole) {
                auto* address = ::mmap(nullptr, imported->extent, PROT_NONE,
                                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                expect(address != MAP_FAILED,
                       "reserve an independently free frontend mapping range");
                reply.address = reinterpret_cast<std::uintptr_t>(address);
                expect(::munmap(address, imported->extent) == 0,
                       "publication address has no server mapping to overwrite");
                ++host_imports;
            } else {
                reply.address = 0x400000;
                ++staging_imports;
            }
            reply.id = imported->resource;
            imports.push_back(std::move(imported));
            return reply;
        }
        if (opcode == ExecutionOpcode::kReleaseHostMemory) {
            std::uint64_t resource{};
            expect(reader.read_u64(resource) && reader.empty(), "exact host cleanup payload");
            const auto imported = std::ranges::find_if(imports, [&](const auto& item) {
                return item->helper == received.connection && item->resource == resource;
            });
            expect(imported != imports.end() && (*imported)->mapping != MAP_FAILED,
                   "cleanup selects only one exact live imported allocation");
            auto& memory = **imported;
            if (memory.origin) {
                expect(effective->continuation &&
                           effective->continuation->flags == kExecutionSourceContinuationCleanup,
                       "imported group resources never borrow ambient cleanup TLS");
                const auto& scope = *effective->continuation;
                expect(
                    scope.caller == memory.origin->caller && scope.group == memory.origin->group &&
                        scope.origin_source == memory.origin->origin_source &&
                        scope.cleanup_parent == memory.origin->ordinary_sequence &&
                        scope.ordinary_sequence == memory.next_cleanup++,
                    "durable cleanup binds old caller/group/origin/import and independent ordinal");
            }
            expect(::munmap(memory.mapping, memory.extent) == 0,
                   "release exact mock helper backing once");
            memory.mapping = MAP_FAILED;
            if (helper_roles[received.connection] == kExecutionHostOwnerRole)
                ++host_cleanups;
            else
                ++staging_cleanups;
            return reply;
        }
        if (opcode == ExecutionOpcode::kHostTransfer ||
            opcode == ExecutionOpcode::kStagedHostTransfer) {
            std::uint64_t helper{}, resource{}, host_offset{}, allocation{}, device_offset{},
                bytes{}, stream{};
            std::uint32_t direction{};
            expect(reader.read_u64(helper) && reader.read_u64(resource) &&
                       reader.read_u64(host_offset) && reader.read_u64(allocation) &&
                       allocation == 701 && reader.read_u64(device_offset) &&
                       reader.read_u64(bytes) && bytes == 64 && reader.read_u64(stream) &&
                       reader.read_u32(direction) && direction <= 1 && reader.empty() &&
                       !host_offset && !device_offset,
                   "complete exact staged/pinned transfer fields survive ordinary wrapping");
            const auto imported = std::ranges::find_if(imports, [&](const auto& item) {
                return item->helper == helper && item->resource == resource &&
                       item->mapping != MAP_FAILED;
            });
            expect(imported != imports.end(), "transfer uses a live imported helper capability");
            if (direction)
                std::memcpy(device_values.data(), (*imported)->mapping, bytes);
            else
                std::memcpy((*imported)->mapping, device_values.data(), bytes);
            transferred_values += static_cast<unsigned int>(device_values.size());
            return reply;
        }
        if (opcode == ExecutionOpcode::kCaptureFrontier) {
            std::uint32_t kind{};
            std::uint64_t stream{};
            expect(reader.read_u32(kind) && kind <= 2 && reader.read_u64(stream) && reader.empty(),
                   "frontier helper preserves its exact ordinary scope payload");
            reply.id = 811;
            return reply;
        }
        if (opcode == ExecutionOpcode::kQueryFrontier ||
            opcode == ExecutionOpcode::kReleaseFrontier) {
            std::uint64_t frontier{};
            expect(reader.read_u64(frontier) && frontier == 811 && reader.empty(),
                   "query/release preserves exact prior frontier identity");
            return reply;
        }
        if (opcode == ExecutionOpcode::kCreateStream) {
            std::uint32_t flags{};
            expect(reader.read_u32(flags) && reader.empty(), "exact owned stream creation");
            reply.id = next_stream++;
            return reply;
        }
        if (opcode == ExecutionOpcode::kClose || opcode == ExecutionOpcode::kDestroyStream ||
            opcode == ExecutionOpcode::kRelease)
            return reply;
        expect(opcode == ExecutionOpcode::kNcclCall && received.connection == transport,
               "all NCCL operations use one process transport, not the source context");
        std::uint64_t caller_primary_domain{};
        ExecutionNcclRequest request;
        expect(reader.read_u64(caller_primary_domain) &&
                   read_execution_nccl_request(reader, request) && reader.empty(),
               "primary provenance prefix plus canonical typed NCCL request");
        expect(!caller_primary_domain || caller_primary_domain == 91,
               "primary domain provenance is distinct from source-session authority");
        auto& sequence = sequences[request.scope.thread[1]];
        expect(request.scope.thread[0] == static_cast<std::uint64_t>(::getpid()) &&
                   request.scope.thread[1] && request.scope.sequence == ++sequence,
               "nonrecycled caller incarnation and dense sent-RPC sequence");
        ++operations[request.call.index()];
        ExecutionNcclReply envelope;
        envelope.is_accepted = true;
        envelope.operation = next_operation++;
        envelope.original_status = 0;
        if (const auto* uid = std::get_if<ExecutionNcclGetUniqueId>(&request.call)) {
            if (uid->has_output)
                envelope.observation = uid_write(std::byte{0x5a});
            else
                envelope.original_status = ncclInvalidArgument;
        } else if (std::holds_alternative<ExecutionNcclInitRank>(request.call) ||
                   std::holds_alternative<ExecutionNcclInitRankConfig>(request.call)) {
            const auto* direct = std::get_if<ExecutionNcclInitRank>(&request.call);
            const auto* configured = std::get_if<ExecutionNcclInitRankConfig>(&request.call);
            const auto& init = direct ? *direct : configured->initialization;
            expect(init.ranks == 1 && init.rank == 0, "original rank scalars survive");
            if (configured && configured->config) {
                const auto& config = *configured->config;
                expect(config.declared_bytes == 88 && config.net_name == "Socket" &&
                           config.comm_name == "frontend-owned-name",
                       "config owns both caller strings");
            }
            if (init.has_output) {
                envelope.reserved_communicator = {91, next_comm++};
                if (request.scope.group && !publishes_group_init_immediately)
                    pending.push_back({request.scope, envelope.reserved_communicator});
                else
                    envelope.observation = comm_write(envelope.reserved_communicator);
            } else
                envelope.original_status = ncclInvalidArgument;
        } else if (const auto* query = std::get_if<ExecutionNcclGetAsyncError>(&request.call)) {
            if (query->has_output) {
                envelope.observation.effects = 4;
                envelope.observation.async_error_mask = 15;
                envelope.observation.async_error = -1234567;
            } else
                envelope.original_status = ncclInvalidArgument;
        } else if (std::holds_alternative<ExecutionNcclGroupStart>(request.call)) {
            expect(request.scope.group != 0, "first Start reserves nonzero group incarnation");
            ++depths[request.scope.thread[1]];
            groups[request.scope.thread[1]] = request.scope.group;
        } else if (std::holds_alternative<ExecutionNcclGroupEnd>(request.call)) {
            auto& depth = depths[request.scope.thread[1]];
            expect(depth && request.scope.group, "End authenticates existing group incarnation");
            --depth;
            // Publish only the original exact scopes. An outer group owns these
            // destinations even after the inner wrappers have returned.
            for (const auto& binding : pending)
                if (!depth && binding.first.group == request.scope.group)
                    envelope.publications.push_back({binding.first, comm_write(binding.second)});
            if (!depth)
                std::erase_if(pending, [&](const auto& binding) {
                    return binding.first.group == request.scope.group;
                });
            if (!envelope.publications.empty())
                envelope.context_effect = {ExecutionNcclContextEffectKind::kSelectPrimary, 0, 91};
        } else {
            const ExecutionNcclAllGather* collective =
                std::get_if<ExecutionNcclAllGather>(&request.call);
            if (const auto* reduce = std::get_if<ExecutionNcclAllReduce>(&request.call)) {
                collective = &reduce->collective;
                expect(reduce->reduction == ncclSum, "raw reduction bits survive");
            }
            ExecutionNcclStream stream;
            ExecutionNcclBuffer buffer;
            if (collective) {
                expect(collective->count == 4 && collective->datatype == ncclFloat32,
                       "collective count and raw datatype survive");
                stream = collective->stream;
                buffer = collective->receive;
                expect(collective->send.source == source && collective->send.allocation == 701 &&
                           collective->send.offset == 4,
                       "send is authenticated allocation plus offset");
                if (operations[8] == 1 && !operations[9] && !operations[10])
                    expect(
                        caller_primary_domain == 91 && !request.scope.context,
                        "projected primary without source session is not encoded as native NULL");
            } else if (const auto* bcast = std::get_if<ExecutionNcclBcast>(&request.call)) {
                expect(bcast->count == 4 && bcast->root == 0, "broadcast scalars survive");
                stream = bcast->stream;
                buffer = bcast->buffer;
            } else
                goto encoded;
            expect(buffer.source == source && buffer.allocation == 701 && buffer.offset == 16,
                   "receive never carries caller raw address bits");
            if (stream.kind == ExecutionNcclStreamKind::kOwned) {
                expect(stream.owner == source && stream.object >= 901,
                       "exact explicit/PTDS proxy owner");
                ++owned_streams;
            } else if (stream.kind == ExecutionNcclStreamKind::kDefault)
                ++defaults;
            else if (stream.kind == ExecutionNcclStreamKind::kLegacy)
                ++legacies;
            else
                expect(false, "PTDS must resolve to the existing owned proxy");
        }
    encoded:
        reply.id = envelope.operation;
        reply.original_status = envelope.original_status;
        ExecutionPayloadWriter writer;
        expect(write_execution_nccl_reply(writer, envelope), "encode full canonical NCR1 envelope");
        reply.data = writer.take();
        return reply;
    }
};
void deliver(ExecutionSocketServer& server, ExecutionReceivedRequest& request,
             const ExecutionWireReply& reply) {
    const auto encoded = encode_execution_reply(reply);
    expect(encoded.has_value(), "encode authenticated mock reply");
    auto header = request.message.header;
    header.kind = ExecutionMessageKind::kReply;
    header.session = request.connection;
    header.incarnation = {37, 89};
    expect(server.reply(request.connection, {header, *encoded}),
           "deliver exact request-correlated reply");
}
template <typename Client, typename Response>
void transport(Client&& client, Response&& response) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "read actual peer identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-nccl-frontend-" + std::to_string(::getpid()) + ".sock";
    config.max_peers = 8;
    config.max_pending_requests = 8;
    configure(config.socket_path, *identity);
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start genuine authenticated socket transport");
    auto future = std::async(std::launch::async, std::forward<Client>(client));
    const auto deadline = Clock::now() + std::chrono::seconds{15};
    while (future.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded real socket progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "actual transport authenticates peer identity");
            auto reply = [&] {
                if constexpr (std::is_invocable_v<Response&, ExecutionSocketServer&,
                                                  ExecutionReceivedRequest&>)
                    return response(server, *request);
                else
                    return response(*request);
            }();
            if constexpr (std::is_same_v<decltype(reply), std::optional<ExecutionWireReply>>) {
                if (reply) deliver(server, *request, *reply);
            } else {
                deliver(server, *request, reply);
            }
        }
    }
    future.get();
    server.stop();
}
void continuation_lifetimes(const char* provider, bool uses_cold_primary) {
    Mock mock;
    std::vector<ExecutionReceivedRequest> deferred_cleanup;
    std::atomic<bool> cleanup_held{};
    transport(
        [&] {
            lookup(provider);
            CUcontext source{};
            if (!uses_cold_primary)
                expect(cuCtxCreate_v2(&source, 0, 0) == CUDA_SUCCESS,
                       "create warm ordinary source before opening native group");
            expect(ncclGroupStart() == ncclSuccess,
                   "begin exact caller group before ordinary work");
            if (uses_cold_primary)
                expect(cuDevicePrimaryCtxRetain(&source, 0) == CUDA_SUCCESS &&
                           cuCtxSetCurrent(source) == CUDA_SUCCESS,
                       "cold source Hello is itself a continuation, not an invented prior source");
            CUdeviceptr device{};
            mock.rejects_next_allocate = true;
            expect(cuMemAlloc_v2(&device, 64) == CUDA_ERROR_OUT_OF_MEMORY &&
                       cuMemAlloc_v2(&device, 64) == CUDA_SUCCESS,
                   "clean wrapped rejection consumes one normal ordinal and permits next original");
            void* pinned{};
            expect(cuMemHostAlloc(&pinned, 64, 0) == CUDA_SUCCESS && pinned,
                   "HST1 bootstrap and real FD import carry the original caller/source");
            std::array<std::uint32_t, 16> first{}, later{};
            for (std::size_t index = 0; index != first.size(); ++index) {
                first[index] = static_cast<std::uint32_t>(index * 17 + 11);
                later[index] = static_cast<std::uint32_t>(index * 19 + 13);
            }
            std::memcpy(pinned, first.data(), sizeof(first));
            expect(cuMemcpyHtoDAsync_v2(device, pinned, sizeof(first), nullptr) == CUDA_SUCCESS,
                   "pinned host capability survives complete wrapped transfer");
            struct Guarded {
                std::array<std::uint32_t, 4> before{};
                std::array<std::uint32_t, 16> values{};
                std::array<std::uint32_t, 4> after{};
            } output;
            output.before.fill(0xabcddcba);
            output.after.fill(0x12344321);
            expect(cuMemcpyHtoDAsync_v2(device, later.data(), sizeof(later), nullptr) ==
                           CUDA_SUCCESS &&
                       cuMemcpyDtoH_v2(output.values.data(), device, sizeof(output.values)) ==
                           CUDA_SUCCESS &&
                       output.values == later &&
                       std::ranges::all_of(output.before,
                                           [](auto value) { return value == 0xabcddcba; }) &&
                       std::ranges::all_of(output.after,
                                           [](auto value) { return value == 0x12344321; }) &&
                       std::memcmp(pinned, first.data(), sizeof(first)) == 0,
                   "cold STG1/import/transfer preserves every value and all independent caller "
                   "guards");
            const auto deadline = Clock::now() + std::chrono::seconds{5};
            while (!cleanup_held.load()) {
                expect(Clock::now() < deadline, "background cleanup arrives before outer End");
                std::this_thread::yield();
            }
            expect(cuCtxSynchronize() == CUDA_SUCCESS && ncclGroupEnd() == ncclSuccess,
                   "held independent cleanup never consumes or blocks normal ordinals and End");
            auto other = std::async(std::launch::async, [pinned] {
                expect(cuMemFreeHost(pinned) == CUDA_SUCCESS,
                       "different thread without NCCL TLS uses durable HST1 import cleanup parent");
            });
            other.get();
            expect(cuMemFree_v2(device) == CUDA_SUCCESS,
                   "ordinary device allocation remains live until explicit release");
            if (uses_cold_primary)
                expect(cuDevicePrimaryCtxRelease_v2(0) == CUDA_SUCCESS,
                       "primary close waits exact retained staging cleanup after End");
            else
                expect(cuCtxDestroy_v2(source) == CUDA_SUCCESS,
                       "source close waits exact retained staging cleanup after End");
            ncclUniqueId id{};
            expect(
                ncclGetUniqueId(&id) == ncclSuccess && !glimmer_nccl_frontend_native_calls(),
                "ordinary and cleanup namespaces never consume NCCL public sequence or fallback");
        },
        [&](ExecutionSocketServer& server,
            ExecutionReceivedRequest& request) -> std::optional<ExecutionWireReply> {
            const auto view = normalize_execution_request(request.message.header.operation,
                                                          request.message.payload);
            expect(view.has_value(),
                   "continuation fixture receives a canonical effective operation");
            if (uses_cold_primary && view->operation == ExecutionOpcode::kHello &&
                decode_execution_context_bootstrap(view->payload))
                expect(view->continuation && !view->continuation->origin_source,
                       "cold primary source Hello explicitly has no prior source session");
            if (view->continuation &&
                view->continuation->flags == kExecutionSourceContinuationCleanup &&
                mock.depths[view->continuation->caller[1]]) {
                expect(deferred_cleanup.size() < 2,
                       "bounded independently owned staging cleanup requests");
                deferred_cleanup.push_back(std::move(request));
                cleanup_held = true;
                return std::nullopt;
            }
            auto reply = mock.respond(request);
            for (std::size_t index = 0; index < deferred_cleanup.size();) {
                auto& cleanup = deferred_cleanup[index];
                const auto pending = normalize_execution_request(cleanup.message.header.operation,
                                                                 cleanup.message.payload);
                expect(pending && pending->continuation,
                       "held cleanup retains its complete immutable scope");
                if (mock.depths[pending->continuation->caller[1]]) {
                    ++index;
                    continue;
                }
                deliver(server, cleanup, mock.respond(cleanup));
                deferred_cleanup.erase(deferred_cleanup.begin() +
                                       static_cast<std::ptrdiff_t>(index));
            }
            return reply;
        });
    expect(deferred_cleanup.empty() && mock.clean_rejections == 1 && mock.host_imports == 1 &&
               mock.staging_imports == 2 && mock.host_cleanups == 1 && mock.staging_cleanups == 2 &&
               mock.transferred_values == 48 && mock.ordinary_calls >= 10 &&
               mock.ordinary_sequences.size() == 1 && mock.operations[6] == 1 &&
               mock.operations[7] == 1 && mock.operations[0] == 1 &&
               std::ranges::all_of(
                   mock.imports, [](const auto& memory) { return memory->mapping == MAP_FAILED; }),
           "complete HST1/STG1 parents retire once across End without ambient thread authority");
}
void continuation_uncertainty(const char* provider) {
    Mock mock;
    unsigned int wrapped_calls{};
    transport(
        [&] {
            lookup(provider);
            CUcontext source{};
            expect(cuCtxCreate_v2(&source, 0, 0) == CUDA_SUCCESS && ncclGroupStart() == ncclSuccess,
                   "establish one exact ordinary source and native group before carrier loss");
            CUdeviceptr device{};
            expect(cuMemAlloc_v2(&device, 64) == CUDA_ERROR_UNKNOWN &&
                       cuMemAlloc_v2(&device, 64) != CUDA_SUCCESS &&
                       ncclGroupEnd() == ncclSystemError && !glimmer_nccl_frontend_native_calls(),
                   "uncertain ordinary receipt quarantines its caller without replay or fake End");
        },
        [&](ExecutionSocketServer& server,
            ExecutionReceivedRequest& request) -> std::optional<ExecutionWireReply> {
            const auto view = normalize_execution_request(request.message.header.operation,
                                                          request.message.payload);
            expect(view.has_value(), "uncertain ordinary request is canonical before carrier loss");
            if (view->continuation) {
                expect(++wrapped_calls == 1 && view->operation == ExecutionOpcode::kAllocate &&
                           view->continuation->ordinary_sequence == 1,
                       "one canonical normal request is submitted, never a continuation replay");
                (void)mock.respond(request);
                server.close_peer(request.connection);
                return std::nullopt;
            }
            return mock.respond(request);
        });
    expect(wrapped_calls == 1 && mock.operations[6] == 1 && !mock.operations[7],
           "lost host reply cannot publish physical cleanup or another requested native entry");
}
void ordinary(const char* provider) {
    Mock mock;
    transport(
        [&] {
            lookup(provider);
            ncclUniqueId id{};
            expect(ncclGetUniqueId(&id) == ncclSuccess, "UID round trip");
            for (const char value : id.internal) expect(value == 0x5a, "all UID bytes published");
            expect(ncclGetUniqueId(&id) == ncclSuccess,
                   "equal-value native writes remain valid effects");
            expect(ncclGetUniqueId(nullptr) == ncclInvalidArgument,
                   "NULL UID preserves original error");
            CUcontext source{};
            expect(cuCtxCreate_v2(&source, 0, 0) == CUDA_SUCCESS, "create original source context");
            CUstream stream{};
            CUdeviceptr memory{};
            expect(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS &&
                       cuMemAlloc_v2(&memory, 64) == CUDA_SUCCESS,
                   "create exact source resources");
            ncclComm_t first{}, second{}, deferred{};
            expect(ncclCommInitRank(&first, 1, id, 0) == ncclSuccess && first,
                   "direct init returns opaque token");
            ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
            config.netName = "Socket";
            config.commName = "frontend-owned-name";
            expect(ncclCommInitRankConfig(&second, 1, id, 0, &config) == ncclSuccess && second,
                   "complete config is copied, not pointer transported");
            expect(ncclGroupStart() == ncclSuccess && ncclGroupStart() == ncclSuccess,
                   "nested native group depth");
            expect(ncclCommInitRankConfig(&deferred, 1, id, 0, nullptr) == ncclSuccess && !deferred,
                   "grouped init keeps original caller destination deferred");
            expect(ncclGroupEnd() == ncclSuccess && !deferred,
                   "inner End cannot publish grouped initialization early");
            expect(ncclGroupEnd() == ncclSuccess && deferred,
                   "outer End publishes original bound output");
            CUcontext primary{};
            expect(cuCtxGetCurrent(&primary) == CUDA_SUCCESS && primary && primary != source,
                   "only native typed context effect selects virtual primary");
            // Owned opaque SDK address, never dereferenced by this caller.
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            auto* send = reinterpret_cast<void*>(static_cast<std::uintptr_t>(memory + 4));
            // Owned opaque SDK address, never dereferenced by this caller.
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            auto* receive = reinterpret_cast<void*>(static_cast<std::uintptr_t>(memory + 16));
            expect(ncclAllGather(send, receive, 4, ncclFloat32, first,
                                 reinterpret_cast<cudaStream_t>(stream)) == ncclSuccess,
                   "pre-init source resources work after actual primary selection");
            expect(cuCtxSetCurrent(source) == CUDA_SUCCESS, "restore caller source context");
            expect(ncclAllReduce(send, receive, 4, ncclFloat32, ncclSum, first, nullptr) ==
                           ncclSuccess &&
                       ncclBcast(receive, 4, ncclFloat32, 0, first,
                                 reinterpret_cast<cudaStream_t>(CU_STREAM_LEGACY)) == ncclSuccess &&
                       ncclAllGather(send, receive, 4, ncclFloat32, first,
                                     reinterpret_cast<cudaStream_t>(CU_STREAM_PER_THREAD)) ==
                           ncclSuccess,
                   "default, legacy and caller PTDS have distinct exact representations");
            ncclResult_t async{};
            expect(ncclCommGetAsyncError(first, &async) == ncclSuccess,
                   "async query requested status");
            std::int32_t raw{};
            std::memcpy(&raw, &async, sizeof(raw));
            expect(raw == -1234567,
                   "async output preserves all raw signed bits without enum evaluation");
            expect(ncclCommGetAsyncError(first, nullptr) == ncclInvalidArgument,
                   "NULL async output reaches original validation");
            expect(ncclCommDestroy(first) == ncclSuccess && ncclCommAbort(second) == ncclSuccess &&
                       ncclCommDestroy(deferred) == ncclSuccess,
                   "both destructive ABIs remain remote");
            expect(ncclCommDestroy(first) == ncclInvalidArgument,
                   "retired opaque handle never native fallback");
            auto other = std::async(std::launch::async, [] {
                ncclUniqueId other_id{};
                expect(ncclGetUniqueId(&other_id) == ncclSuccess,
                       "second stable caller uses same transport");
            });
            other.get();
            const auto child = ::fork();
            expect(child >= 0, "fork inherited frontend state");
            if (!child) ::_exit(ncclGroupStart() == ncclInvalidUsage ? 0 : 1);
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "fork rejects before inherited mutex or transport");
            expect(!glimmer_nccl_frontend_native_calls(),
                   "no caller-provider callback ever executes");
        },
        [&](ExecutionReceivedRequest& request) { return mock.respond(request); });
    expect(mock.nccl_hellos == 1 && mock.sequences.size() == 2 &&
               std::ranges::all_of(mock.operations, [](auto count) { return count != 0; }) &&
               mock.owned_streams == 2 && mock.defaults == 1 && mock.legacies == 1,
           "all eleven APIs, two caller incarnations and four stream controls observed");
}
void hostile(const char* provider, unsigned int which) {
    Mock mock;
    unsigned int requests{};
    transport(
        [&] {
            lookup(provider);
            CUcontext lower{}, upper{};
            if (which == 14)
                expect(cuCtxCreate_v2(&lower, 0, 0) == CUDA_SUCCESS &&
                           cuCtxCreate_v2(&upper, 0, 0) == CUDA_SUCCESS,
                       "establish nested virtual stack before hostile current-NULL projection");
            ncclUniqueId id{};
            expect(ncclGetUniqueId(&id) == ncclSuccess, "healthy authenticated prefix");
            std::memset(&id, 0x33, sizeof(id));
            ncclComm_t communicator{};
            ncclResult_t async{};
            if (which >= 17)
                expect(ncclCommInitRank(&communicator, 1, id, 0) == ncclSuccess && communicator,
                       "establish original communicator for output-completeness control");
            const auto result = which >= 17 ? ncclCommGetAsyncError(communicator, &async)
                                            : ncclGetUniqueId(which == 11 ? nullptr : &id);
            // Once the original status is present, later proof faults quarantine but
            // do not invent another requested native status or replay the call.
            const auto expected = which == 7   ? ncclSystemError
                                  : which == 9 ? ncclInvalidUsage
                                               : ncclSuccess;
            expect(result == expected, "native status survives later wire/effect proof fault");
            for (const char value : id.internal)
                expect(value == 0x33, "malformed reply publishes no UID bytes");
            std::int32_t raw_async{};
            std::memcpy(&raw_async, &async, sizeof(raw_async));
            expect(raw_async == 0, "incomplete successful scalar publishes no bytes");
            if (which == 14) {
                auto observed = lower;
                expect(cuCtxGetCurrent(&observed) == CUDA_ERROR_UNKNOWN && observed == lower,
                       "unproved NULL effect does not silently pop to the healthy lower context");
            }
            expect(ncclGetUniqueId(&id) == ncclSystemError,
                   "uncertain process transport cannot replay/reopen");
            expect(!glimmer_nccl_frontend_native_calls(),
                   "hostile envelope has no native fallback");
        },
        [&](ExecutionReceivedRequest& request) {
            auto reply = mock.respond(request);
            if (request.message.header.operation != ExecutionOpcode::kNcclCall ||
                ++requests != (which >= 17 ? 3U : 2U))
                return reply;
            ExecutionPayloadReader reader(reply.data);
            ExecutionNcclReply envelope;
            expect(read_execution_nccl_reply(reader, envelope) && reader.empty(),
                   "decode initial healthy envelope");
            envelope.observation = {};
            switch (which) {
                case 0:
                    ++reply.id;
                    break;
                case 1:
                    envelope.original_status = ncclInvalidArgument;
                    break;
                case 2:
                    reply.parameter_bytes = 2;
                    break;
                case 3:
                    reply.data.push_back(std::byte{});
                    return reply;
                case 4:
                    envelope.observation = comm_write({91, 77});
                    break;
                case 5:
                    envelope.context_effect = {ExecutionNcclContextEffectKind::kSelectPrimary, 1,
                                               91};
                    break;
                case 6:
                    reply.address = 19;
                    break;
                case 7:
                    reply.disposition = ExecutionWireDisposition::kRejected;
                    reply.original_status.reset();
                    reply.id = 0;
                    envelope = {};
                    break;
                case 8:
                    reply.disposition = ExecutionWireDisposition::kRejected;
                    break;
                case 9:
                    reply.status = CUDA_ERROR_UNKNOWN;
                    reply.original_status = ncclInvalidUsage;
                    envelope.original_status = ncclInvalidUsage;
                    reply.parameter_bytes = 1;
                    break;
                case 10:
                    reply.status = CUDA_ERROR_UNKNOWN;
                    reply.parameter_bytes = 1;
                    break;
                case 11:
                    reply.original_status = 0;
                    envelope.original_status = 0;
                    envelope.observation = uid_write(std::byte{0x5a});
                    break;
                case 12:
                    envelope.publications.push_back({{{1, 99}, 1, 7, 0}, comm_write({91, 1})});
                    break;
                case 13:
                    reply.parameter_bytes = 1;
                    reply.arguments.push_back({0, 1});
                    break;
                case 14:
                    envelope.context_effect.kind = static_cast<ExecutionNcclContextEffectKind>(2);
                    break;
                case 15:
                    break;  // Canonical absent UID output on requested success.
                case 16:
                    envelope.observation.effects = 1;
                    envelope.observation.unique_id_mask[0] = std::byte{1};
                    envelope.observation.unique_id[0] = std::byte{0x5a};
                    break;
                case 17:
                    break;  // Canonical absent async output on requested success.
                case 18:
                    envelope.observation.effects = 4;
                    envelope.observation.async_error_mask = 1;
                    envelope.observation.async_error = 7;
                    break;
                default:
                    expect(false, "known hostile mode");
            }
            ExecutionPayloadWriter writer;
            expect(write_execution_nccl_reply(writer, envelope),
                   "encode structurally canonical hostile envelope");
            reply.data = writer.take();
            return reply;
        });
    expect(requests == (which >= 17 ? 3U : 2U),
           "quarantine does not send a recovery or hidden SDK query");
}
void grouped_publication(const char* provider, unsigned int which) {
    Mock mock;
    unsigned int ends{};
    transport(
        [&] {
            lookup(provider);
            ncclUniqueId id{};
            ncclComm_t first{}, second{};
            auto* page =
                ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            expect(page != MAP_FAILED, "map separately revocable deferred destination");
            auto* other = which == 4 ? static_cast<ncclComm_t*>(page) : &second;
            expect(ncclGroupStart() == ncclSuccess &&
                       ncclCommInitRank(&first, 1, id, 0) == ncclSuccess &&
                       ncclCommInitRank(other, 1, id, 0) == ncclSuccess && !first && !*other,
                   "both exact deferred destinations remain untouched before End");
            if (which == 4)
                expect(::mprotect(page, 4096, PROT_READ) == 0, "revoke output write permission");
            expect(ncclGroupEnd() == ncclSuccess && !first && !*other,
                   "whole publication vector validates before even the first valid destination "
                   "writes");
            expect(ncclGroupEnd() == ncclSystemError,
                   "invalid publication permanently retains group owner");
            expect(::munmap(page, 4096) == 0 && !glimmer_nccl_frontend_native_calls(),
                   "caller lifetime guard has no SDK fallback");
        },
        [&](ExecutionReceivedRequest& request) {
            auto reply = mock.respond(request);
            if (request.message.header.operation != ExecutionOpcode::kNcclCall) return reply;
            ExecutionPayloadReader request_reader(request.message.payload);
            ExecutionNcclRequest call;
            std::uint64_t caller_primary_domain{};
            expect(request_reader.read_u64(caller_primary_domain) &&
                       read_execution_nccl_request(request_reader, call),
                   "decode publication control request");
            if (!std::holds_alternative<ExecutionNcclGroupEnd>(call.call)) return reply;
            ++ends;
            ExecutionPayloadReader reader(reply.data);
            ExecutionNcclReply envelope;
            expect(read_execution_nccl_reply(reader, envelope) && envelope.publications.size() == 2,
                   "actual two full-scope pending output bindings");
            if (which == 0)
                envelope.publications[1].scope = envelope.publications[0].scope;
            else if (which == 1)
                ++envelope.publications[1].scope.sequence;
            else if (which == 2)
                ++envelope.publications[1].scope.context;
            else if (which == 3)
                ++envelope.publications[1].observation.communicator.object;
            else if (which == 5)
                envelope.publications.clear();
            else if (which == 6)
                envelope.publications.pop_back();
            ExecutionPayloadWriter writer;
            expect(write_execution_nccl_reply(writer, envelope),
                   "encode hostile deferred publication");
            reply.data = writer.take();
            return reply;
        });
    expect(ends == 1, "failed publication causes no repeated End entry");
}
void error_effects(const char* provider) {
    Mock mock;
    unsigned int calls{};
    transport(
        [&] {
            lookup(provider);
            ncclUniqueId id{};
            expect(ncclGetUniqueId(&id) == ncclInvalidArgument,
                   "native error preserves independently proved output write");
            for (const char value : id.internal)
                expect(value == 0x5a, "error-path UID bytes written");
            expect(ncclGetUniqueId(&id) == ncclInvalidUsage,
                   "clean remote policy rejection has no fabricated original NCCL status");
            expect(ncclGetUniqueId(&id) == ncclSuccess,
                   "clean rejection consumes sent caller sequence without poisoning healthy "
                   "transport");
            auto* guard = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            expect(guard != MAP_FAILED, "map inaccessible immediate output");
            expect(ncclGetUniqueId(static_cast<ncclUniqueId*>(guard)) == ncclInvalidArgument &&
                       ::munmap(guard, 4096) == 0,
                   "local invalid output never sends an RPC");
            expect(ncclCommInitRank(nullptr, 1, id, 0) == ncclInvalidArgument,
                   "NULL init destination reaches original unchanged");
            expect(ncclGetUniqueId(&id) == ncclSuccess,
                   "original NULL-argument error keeps transport usable");
            expect(!glimmer_nccl_frontend_native_calls(),
                   "all errors are remote or explicit policy");
        },
        [&](ExecutionReceivedRequest& request) {
            auto reply = mock.respond(request);
            if (request.message.header.operation != ExecutionOpcode::kNcclCall) return reply;
            ++calls;
            if (calls > 2) return reply;
            ExecutionPayloadReader reader(reply.data);
            ExecutionNcclReply envelope;
            expect(read_execution_nccl_reply(reader, envelope), "decode error effect envelope");
            if (calls == 1) {
                reply.original_status = ncclInvalidArgument;
                envelope.original_status = ncclInvalidArgument;
            } else {
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = CUDA_ERROR_NOT_SUPPORTED;
                reply.original_status.reset();
                reply.id = 0;
                envelope = {};
            }
            ExecutionPayloadWriter writer;
            expect(write_execution_nccl_reply(writer, envelope),
                   "encode exact error/native-effect relation");
            reply.data = writer.take();
            return reply;
        });
    expect(calls == 5, "local rejected destination consumes neither transport nor caller sequence");
}
void held_reservations(const char* provider) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual held-reservation peer identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-nccl-held-" + std::to_string(::getpid()) + ".sock";
    config.max_pending_requests = 8;
    configure(config.socket_path, *identity);
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start held bootstrap/init transport");
    Mock mock;
    std::optional<ExecutionReceivedRequest> hello, init;
    std::atomic<bool> hello_held{}, uid_started{}, uid_complete{}, init_held{}, release_init{};
    auto client = std::async(std::launch::async, [&] {
        lookup(provider);
        auto initializing = std::async(std::launch::async, [] {
            ncclUniqueId id{};
            ncclComm_t communicator{};
            expect(ncclCommInitRank(&communicator, 1, id, 0) == ncclSuccess && communicator,
                   "held initialization publishes its own retained opaque entry");
            return communicator;
        });
        const auto deadline = Clock::now() + std::chrono::seconds{10};
        while (!hello_held.load()) {
            expect(Clock::now() < deadline, "bounded held bootstrap observation");
            std::this_thread::yield();
        }
        auto other = std::async(std::launch::async, [&] {
            uid_started = true;
            ncclUniqueId id{};
            expect(ncclGetUniqueId(&id) == ncclSuccess,
                   "other caller completes while init reply is held");
            uid_complete = true;
        });
        while (!uid_complete.load() || !init_held.load()) {
            expect(Clock::now() < deadline, "bounded independent caller progress");
            std::this_thread::yield();
        }
        release_init = true;
        const auto communicator = initializing.get();
        other.get();
        ncclResult_t asynchronous{};
        expect(ncclCommGetAsyncError(communicator, &asynchronous) == ncclSuccess &&
                   ncclCommDestroy(communicator) == ncclSuccess,
               "another wrapper cannot sweep a prepared/in-flight initializer's token");
        expect(!glimmer_nccl_frontend_native_calls(),
               "held ownership has no caller-native fallback");
    });
    const auto deadline = Clock::now() + std::chrono::seconds{15};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded held request progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "held request keeps authenticated ownership");
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                hello = std::move(request);
                hello_held = true;
            } else {
                ExecutionPayloadReader reader(request->message.payload);
                std::uint64_t provenance{};
                ExecutionNcclRequest call;
                expect(reader.read_u64(provenance) && read_execution_nccl_request(reader, call),
                       "held request remains canonical NCCL framing");
                if (std::holds_alternative<ExecutionNcclInitRank>(call.call)) {
                    init = std::move(request);
                    init_held = true;
                } else
                    deliver(server, *request, mock.respond(*request));
            }
        }
        if (hello && uid_started.load()) {
            deliver(server, *hello, mock.respond(*hello));
            hello.reset();
        }
        if (init && release_init.load()) {
            deliver(server, *init, mock.respond(*init));
            init.reset();
        }
    }
    client.get();
    expect(mock.nccl_hellos == 1 && mock.operations[1] == 1 && mock.operations[0] == 1 &&
               mock.operations[3] == 1 && mock.operations[4] == 1,
           "one bootstrap, held init, independent UID and exact retained-handle uses");
    server.stop();
}
void immediate_group(const char* provider) {
    Mock mock;
    mock.publishes_group_init_immediately = true;
    transport(
        [&] {
            lookup(provider);
            ncclUniqueId id{};
            ncclComm_t communicator{};
            expect(ncclGroupStart() == ncclSuccess &&
                       ncclCommInitRank(&communicator, 1, id, 0) == ncclSuccess && communicator,
                   "proved immediate grouped output publishes at its actual native boundary");
            expect(ncclGroupEnd() == ncclSuccess && ncclCommDestroy(communicator) == ncclSuccess,
                   "resolved immediate output requires no fabricated duplicate End publication");
            expect(!glimmer_nccl_frontend_native_calls(),
                   "immediate group remains entirely remote");
        },
        [&](ExecutionReceivedRequest& request) { return mock.respond(request); });
    expect(mock.pending.empty(), "no stale grouped output destination survives actual resolution");
}
void fault_effects(const char* provider, unsigned int which) {
    Mock mock;
    unsigned int requests{};
    transport(
        [&] {
            lookup(provider);
            ncclUniqueId id{};
            expect(ncclGetUniqueId(&id) == ncclSuccess,
                   "healthy authenticated fault-effect prefix");
            std::memset(&id, 0x33, sizeof(id));
            int sentinel{};
            const auto before = reinterpret_cast<ncclComm_t>(&sentinel);
            auto communicator = before;
            const auto result =
                which < 2 ? ncclCommInitRank(&communicator, 1, id, 0) : ncclGetUniqueId(&id);
            expect(result == (which < 2 ? ncclInvalidArgument : ncclSuccess),
                   "original native success/error survives independent ownership fault");
            if (which < 2) {
                expect(!communicator,
                       "proved actual NULL error-path write projects despite fault flag or carrier "
                       "error");
            } else {
                const char expected = which == 2 ? 0x5a : 0x33;
                for (const char value : id.internal)
                    expect(value == expected,
                           "well-formed proved UID writes project; malformed fault envelopes stay "
                           "atomic");
                expect(communicator == before,
                       "fault-effect projection touches only its bound output");
            }
            expect(
                ncclGetUniqueId(&id) == ncclSystemError,
                "projecting proved effects never authorizes replay or clears ownership quarantine");
            expect(!glimmer_nccl_frontend_native_calls(),
                   "fault projection never invokes caller SDK");
        },
        [&](ExecutionReceivedRequest& request) {
            auto reply = mock.respond(request);
            if (request.message.header.operation != ExecutionOpcode::kNcclCall || ++requests != 2)
                return reply;
            ExecutionPayloadReader reader(reply.data);
            ExecutionNcclReply envelope;
            expect(read_execution_nccl_reply(reader, envelope) && reader.empty(),
                   "decode actual fault-effect request-correlated envelope");
            reply.status = which == 0 ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
            reply.parameter_bytes = which == 1 ? 0 : 1;
            if (which < 2) {
                reply.original_status = ncclInvalidArgument;
                envelope.original_status = ncclInvalidArgument;
                envelope.observation = comm_write({});
            } else {
                expect(envelope.original_status == 0 &&
                           envelope.observation == uid_write(std::byte{0x5a}),
                       "actual requested success owns complete UID write effect before later proof "
                       "loss");
                if (which == 3)
                    envelope.publications.push_back({{{71, 73}, 1, 2, 0}, comm_write({91, 77})});
                if (which == 4) reply.status = -1;
            }
            ExecutionPayloadWriter writer;
            expect(write_execution_nccl_reply(writer, envelope),
                   "encode canonical proven fault effects");
            reply.data = writer.take();
            return reply;
        });
    expect(requests == 2, "fault effects produce no hidden recovery/replay request");
}
void isolated(const char* executable, const char* provider, const std::string& scenario) {
    const auto child = ::fork();
    expect(child >= 0, "fork fresh process-isolated frontend scenario");
    if (!child) {
        ::execl(executable, executable, provider, scenario.c_str(), static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fresh frontend scenario exits successfully");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 3, "provider path and optional isolated scenario");
        if (argc == 3) {
            const std::string scenario{argv[2]};
            if (scenario == "ordinary")
                ordinary(argv[1]);
            else if (scenario == "effects")
                error_effects(argv[1]);
            else if (scenario == "held")
                held_reservations(argv[1]);
            else if (scenario == "immediate")
                immediate_group(argv[1]);
            else if (scenario == "continuation")
                continuation_lifetimes(argv[1], false);
            else if (scenario == "continuation-cold")
                continuation_lifetimes(argv[1], true);
            else if (scenario == "continuation-uncertain")
                continuation_uncertainty(argv[1]);
            else if (scenario.starts_with("fault-"))
                fault_effects(argv[1], static_cast<unsigned int>(std::stoul(scenario.substr(6))));
            else if (scenario.starts_with("group-"))
                grouped_publication(argv[1],
                                    static_cast<unsigned int>(std::stoul(scenario.substr(6))));
            else
                hostile(argv[1], static_cast<unsigned int>(std::stoul(scenario)));
        } else {
            isolated(argv[0], argv[1], "ordinary");
            isolated(argv[0], argv[1], "effects");
            isolated(argv[0], argv[1], "held");
            isolated(argv[0], argv[1], "immediate");
            isolated(argv[0], argv[1], "continuation");
            isolated(argv[0], argv[1], "continuation-cold");
            isolated(argv[0], argv[1], "continuation-uncertain");
            for (unsigned int which = 0; which != 5; ++which)
                isolated(argv[0], argv[1], "fault-" + std::to_string(which));
            for (unsigned int which = 0; which != 7; ++which)
                isolated(argv[0], argv[1], "group-" + std::to_string(which));
            for (unsigned int which = 0; which != kHostileCases; ++which)
                isolated(argv[0], argv[1], std::to_string(which));
        }
    } catch (const std::exception& error) {
        std::cerr << (argc == 3 ? argv[2] : "suite") << ": " << error.what() << '\n';
        return 1;
    }
    return 0;
}
