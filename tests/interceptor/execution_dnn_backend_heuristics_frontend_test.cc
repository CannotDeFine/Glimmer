#include "execution_dnn_profile_fixture.h"
#include "glimmer/control/execution_dnn_backend_protocol.h"
#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" unsigned int glimmer_dnn_backend_graph_native_calls();

namespace {
using namespace glimmer::control;
using Native = cudnnBackendDescriptor_t;
using Direction = ExecutionDnnBackendAttributeDirection;
using Operation = ExecutionDnnBackendAttributeOperation;
constexpr std::uint32_t kAvailable = 3;
constexpr std::int64_t kMarker = -719;

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void success(cudnnStatus_t status) {
    expect(status == CUDNN_STATUS_SUCCESS, "synthetic public backend success");
}
struct Api {
    decltype(&cudnnBackendCreateDescriptor) create{};
    decltype(&cudnnBackendDestroyDescriptor) destroy{};
    decltype(&cudnnBackendInitialize) initialize{};
    decltype(&cudnnBackendFinalize) finalize{};
    decltype(&cudnnBackendSetAttribute) set{};
    decltype(&cudnnBackendGetAttribute) get{};
};
Api lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "open existing synthetic graph provider");
    Api api;
#define LOOKUP(member, symbol)                                        \
    do {                                                              \
        auto* address = ::dlsym(library, #symbol);                    \
        expect(address == reinterpret_cast<void*>(&(symbol)) &&       \
                   address == ::dlsym(RTLD_DEFAULT, #symbol),         \
               "direct/global/explicit graph lookup is intercepted"); \
        api.member = reinterpret_cast<decltype(api.member)>(address); \
    } while (false)
    LOOKUP(create, cudnnBackendCreateDescriptor);
    LOOKUP(destroy, cudnnBackendDestroyDescriptor);
    LOOKUP(initialize, cudnnBackendInitialize);
    LOOKUP(finalize, cudnnBackendFinalize);
    LOOKUP(set, cudnnBackendSetAttribute);
    LOOKUP(get, cudnnBackendGetAttribute);
#undef LOOKUP
    expect(!::dlclose(library), "cached public functions survive explicit provider close");
    return api;
}
Native create(const Api& api, unsigned kind) {
    Native value{};
    success(api.create(static_cast<cudnnBackendDescriptorType_t>(kind), &value));
    expect(value, "non-null metadata token");
    return value;
}

class GuardedArray {
   public:
    GuardedArray() {
        const auto size = ::sysconf(_SC_PAGESIZE);
        expect(size > 0, "page size");
        page_ = static_cast<std::size_t>(size);
        memory_ = ::mmap(nullptr, page_ * 2, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(memory_ != MAP_FAILED, "guarded destination token array");
    }
    ~GuardedArray() {
        if (::munmap(memory_, page_ * 2)) std::terminate();
    }
    GuardedArray(const GuardedArray&) = delete;
    GuardedArray& operator=(const GuardedArray&) = delete;
    void* store(std::span<const Native> values) {
        expect(!::mprotect(memory_, page_, PROT_READ | PROT_WRITE), "writable setup page");
        auto* result = static_cast<std::byte*>(memory_) + page_ - values.size_bytes();
        if (!values.empty()) std::memcpy(result, values.data(), values.size_bytes());
        expect(!::mprotect(memory_, page_, PROT_READ), "descriptor token array is input-only");
        return result;
    }

   private:
    void* memory_{};
    std::size_t page_{};
};

struct Family {
    unsigned source_kind{}, name{}, target_kind{};
};
constexpr std::array<Family, 3> kFamilies{{{4, 202, 3}, {2, 1302, 8}, {3, 302, 7}}};
std::uint32_t prefix(Family family, std::int64_t requested, bool has_array) {
    if (!has_array || requested < 0) return 0;
    if (family.name == 302) return static_cast<std::uint32_t>(requested);
    if (family.name == 1302 && requested == 0) return kAvailable;
    return static_cast<std::uint32_t>(std::min<std::int64_t>(requested, kAvailable));
}
std::int64_t count(Family family, std::int64_t requested, bool has_array) {
    if (family.name == 202) return has_array ? std::min<std::int64_t>(requested, 2) : kAvailable;
    if (family.name == 1302 && requested == 0) return kAvailable;
    return std::clamp<std::int64_t>(requested, 0, kAvailable);
}

void query(const Api& api, Native source, Family family, std::int64_t requested,
           std::span<const Native> targets, bool has_count = true) {
    GuardedArray guard;
    auto* array = guard.store(targets);
    const std::vector<Native> before(targets.begin(), targets.end());
    auto actual = kMarker;
    const auto status =
        api.get(source, static_cast<cudnnBackendAttributeName_t>(family.name),
                CUDNN_TYPE_BACKEND_DESCRIPTOR, requested, has_count ? &actual : nullptr, array);
    if (requested < 0)
        expect(status == CUDNN_STATUS_BAD_PARAM && actual == kMarker,
               "negative requested capacity reaches original error without reading PROT_NONE");
    else {
        success(status);
        expect(actual == (has_count ? count(family, requested, true) : kMarker),
               "original reported count is independent from captured token capacity");
    }
    expect(before.empty() || !std::memcmp(array, before.data(), before.size() * sizeof(Native)),
           "caller descriptor tokens remain unchanged, including unwritten tail positions");
}

void application(const char* provider) {
    const auto api = lookup(provider);
    std::array<Native, 3> sources;
    for (std::size_t i = 0; i != sources.size(); ++i)
        sources[i] = create(api, kFamilies[i].source_kind);
    expect(api.finalize(sources[0]) == CUDNN_STATUS_NOT_INITIALIZED,
           "heuristic Finalize needs execution authority before any request");
    CUcontext context{};
    expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS, "open synthetic owned context");
    const auto graph = create(api, 15);
    const auto mode = CUDNN_HEUR_MODE_INSTANT;
    success(api.set(sources[0], CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode));
    success(api.set(sources[0], CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                    CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph));
    for (const auto source : sources) success(api.finalize(source));

    for (std::size_t index = 0; index != kFamilies.size(); ++index) {
        const auto family = kFamilies[index];
        const auto source = sources[index];
        std::array<Native, 5> targets;
        for (auto& target : targets) target = create(api, family.target_kind);
        for (const auto requested : {INT64_MIN, std::int64_t{-1}, std::int64_t{0}, std::int64_t{1},
                                     std::int64_t{3}, std::int64_t{5}, INT64_MAX}) {
            auto actual = kMarker;
            success(api.get(source, static_cast<cudnnBackendAttributeName_t>(family.name),
                            CUDNN_TYPE_BACKEND_DESCRIPTOR, requested, &actual, nullptr));
            expect(actual == count(family, requested, false),
                   "count-only request preserves raw signed and huge capacities");
        }
        expect(
            api.get(source, static_cast<cudnnBackendAttributeName_t>(family.name),
                    CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, nullptr, nullptr) == CUDNN_STATUS_BAD_PARAM,
            "both absent outputs preserve original native error");
        for (const auto requested : {std::int64_t{-1}, std::int64_t{0}, std::int64_t{1},
                                     std::int64_t{2}, std::int64_t{3}, std::int64_t{5}}) {
            for (const auto target : targets) success(api.initialize(target));
            query(api, source, family, requested,
                  std::span{targets}.first(prefix(family, requested, true)));
        }
        for (const auto target : targets) success(api.initialize(target));
        query(api, source, family, 3, std::span{targets}.first(3), false);
        if (family.name != 302) {
            for (const auto target : targets) success(api.initialize(target));
            query(api, source, family, INT64_MAX, std::span{targets}.first(3));
        } else {
            GuardedArray guard;
            auto actual = kMarker;
            expect(api.get(source, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                           INT64_MAX, &actual, guard.store({})) == CUDNN_STATUS_BAD_PARAM &&
                       actual == kMarker,
                   "excessive native all-target validation span is bounded before caller read");
        }
        for (const auto target : targets) success(api.initialize(target));
        const std::array aliases{targets[0], targets[1], targets[0]};
        if (family.name == 202) {
            const std::array repeated{targets[0], targets[0]};
            const auto before = repeated;
            auto actual = kMarker;
            expect(
                api.get(source, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 2,
                        &actual, const_cast<Native*>(repeated.data())) == CUDNN_STATUS_BAD_PARAM &&
                    actual == kMarker && repeated == before,
                "second generated Config alias fails after the first mutation without count "
                "or token publication");
            success(api.initialize(targets[0]));
            query(api, source, family, 1, std::span{targets}.first(1));
        } else {
            query(api, source, family, 3, aliases);
            if (family.name == 302) {
                auto wrong_tail = targets;
                wrong_tail.back() = source;
                auto actual = kMarker;
                expect(api.get(source, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                               CUDNN_TYPE_BACKEND_DESCRIPTOR, 5, &actual,
                               wrong_tail.data()) == CUDNN_STATUS_BAD_PARAM &&
                           actual == kMarker && wrong_tail.back() == source,
                       "Config validates requested tail targets even though it emits only three");
            }
        }
        const std::array wrong{source};
        auto actual = kMarker;
        expect(api.get(source, static_cast<cudnnBackendAttributeName_t>(family.name),
                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &actual,
                       const_cast<Native*>(wrong.data())) == CUDNN_STATUS_BAD_PARAM &&
                   actual == kMarker && wrong[0] == source,
               "source-as-wrong-target alias terminates without self-pin deadlock");
        auto stale = targets.back();
        success(api.destroy(stale));
        actual = kMarker;
        expect(api.get(source, static_cast<cudnnBackendAttributeName_t>(family.name),
                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &actual,
                       &stale) == CUDNN_STATUS_NOT_INITIALIZED &&
                   actual == kMarker,
               "stale destination rejects locally before final native request");
        for (const auto target : std::span{targets}.first(4)) success(api.destroy(target));
    }
    expect(cuCtxDestroy(context) == CUDA_SUCCESS, "context-first cleanup");
    for (const auto source : sources) {
        success(api.initialize(source));
        success(api.destroy(source));
    }
    success(api.destroy(graph));
    expect(!glimmer_dnn_backend_graph_native_calls(), "no application-process SDK fallback");
}

ExecutionWireReply complete(std::optional<std::int32_t> original = CUDNN_STATUS_SUCCESS) {
    ExecutionWireReply result;
    result.disposition = ExecutionWireDisposition::kComplete;
    result.original_status = original;
    result.status = original.value_or(0);
    return result;
}
struct Node {
    unsigned kind{};
    bool finalized{};
    std::int64_t value{kMarker};
};
struct ExpectedRequest {
    std::int64_t requested{};
    std::uint32_t presence{};
};
std::vector<ExpectedRequest> expected_requests(std::size_t family) {
    std::vector<ExpectedRequest> result;
    for (const auto requested : {INT64_MIN, std::int64_t{-1}, std::int64_t{0}, std::int64_t{1},
                                 std::int64_t{3}, std::int64_t{5}, INT64_MAX})
        result.push_back({requested, 2});
    result.push_back({0, 0});
    for (const auto requested : {std::int64_t{-1}, std::int64_t{0}, std::int64_t{1},
                                 std::int64_t{2}, std::int64_t{3}, std::int64_t{5}})
        result.push_back({requested, 3});
    result.push_back({3, 1});
    result.push_back({INT64_MAX, 3});
    result.push_back({family == 0 ? 2 : 3, 3});
    if (family == 0) result.push_back({1, 3});
    if (family == 2) result.push_back({5, 3});
    result.push_back({1, 3});
    result.push_back({1, 3});
    return result;
}
struct Peer {
    std::uint64_t metadata{}, execution{}, next_id{100};
    std::unordered_map<std::uint64_t, Node> nodes;
    std::array<unsigned, 3> preparations{}, gets{};
    std::array<std::optional<ExecutionDnnBackendAttributeHeader>, 3> pending;
    unsigned setters{}, finalizes{}, duplicate_errors{}, duplicate_successes{};

    ExecutionWireReply attribute(ExecutionOpcode opcode, ExecutionPayloadReader& input) {
        ExecutionDnnBackendAttributeHeader header;
        ExecutionDnnBackendAttribute command;
        Operation operation{};
        if (opcode == ExecutionOpcode::kDnnBackendPrepare) {
            ExecutionDnnBackendPrepare request;
            expect(read_execution_dnn_backend_prepare(input, request) && input.empty(),
                   "exact attribute preparation");
            operation = request.operation;
            header = request.header;
        } else {
            expect(read_execution_dnn_backend_attribute(input, command) && input.empty() &&
                       command.generation == 71,
                   "exact final prepared generation and original header");
            operation =
                opcode == ExecutionOpcode::kDnnBackendSet ? Operation::kSet : Operation::kGet;
            header = command.header;
        }
        const auto kind = nodes.at(header.descriptor).kind;
        if (operation == Operation::kSet) {
            expect(kind == 4 &&
                       ((header.name == 200 && header.type == 8) ||
                        (header.name == 201 && header.type == 15)) &&
                       header.count == 1 && header.presence == 1,
                   "heuristic mode and graph setters retain native ABI");
            if (opcode == ExecutionOpcode::kDnnBackendPrepare) {
                ExecutionPayloadWriter output;
                expect(write_execution_dnn_backend_preparation(output,
                                                               {71, Direction::kInput, 1, 0, 0, 0}),
                       "setter preparation");
                auto result = complete(std::nullopt);
                result.data = output.take();
                return result;
            }
            if (header.name == 201)
                expect(command.values.references.size() == 1 &&
                           command.values.references[0].owner == metadata &&
                           nodes.at(command.values.references[0].id).kind == 15,
                       "graph child retains DNM1 identity");
            else
                expect(command.values.scalars == std::vector<std::byte>(4),
                       "INSTANT mode keeps raw enum zero");
            ++setters;
            return complete();
        }
        const auto found = std::ranges::find_if(kFamilies, [&](auto family) {
            return family.source_kind == kind && family.name == header.name;
        });
        expect(found != kFamilies.end() && header.type == 15, "known generated output family");
        const auto index = static_cast<std::size_t>(found - kFamilies.begin());
        const auto has_array = (header.presence & 1U) != 0;
        const auto has_count = (header.presence & 2U) != 0;
        if (opcode == ExecutionOpcode::kDnnBackendPrepare) {
            const auto sequence = expected_requests(index);
            expect(preparations[index] < sequence.size() && header.descriptor == 100 + index &&
                       header.count == sequence[preparations[index]].requested &&
                       header.presence == sequence[preparations[index]].presence,
                   "Prepare preserves independently scripted signed count and nullable outputs");
            pending[index] = header;
            ++preparations[index];
            if (header.name == 302 && has_array && header.count > 65536) {
                ExecutionWireReply result;
                result.disposition = ExecutionWireDisposition::kRejected;
                result.status = CUDA_ERROR_INVALID_VALUE;
                return result;
            }
            ExecutionPayloadWriter output;
            expect(
                write_execution_dnn_backend_preparation(
                    output, {71, Direction::kDescriptorOutput,
                             prefix(*found, header.count, has_array), 0, has_count ? 8U : 0U, 0}),
                "exact bounded captured prefix independent from emitted output count");
            auto result = complete(std::nullopt);
            result.data = output.take();
            return result;
        }
        expect(pending[index] && header.descriptor == pending[index]->descriptor &&
                   header.name == pending[index]->name && header.type == pending[index]->type &&
                   header.count == pending[index]->count &&
                   header.presence == pending[index]->presence,
               "final Get preserves every original prepared header field without clamping");
        pending[index].reset();
        ++gets[index];
        const auto capacity = prefix(*found, header.count, has_array);
        expect(command.values.scalars.empty() && command.values.references.size() == capacity,
               "final sends only prepared descriptor prefix, never requested-size overread");
        if ((!has_array && !has_count) || (has_array && header.count < 0))
            return complete(CUDNN_STATUS_BAD_PARAM);
        for (const auto& reference : command.values.references)
            expect(reference.kind == ExecutionDnnBackendReferenceKind::kDescriptor &&
                       reference.owner == metadata,
                   "every target belongs to unchanged authenticated metadata owner");
        // A source alias is an intentionally wrong-kind target. This synthetic
        // endpoint returns a recoverable native error and tests the frontend's
        // exclusive-source acquisition, not the backend's typed policy mapping.
        for (const auto& reference : command.values.references)
            if (nodes.at(reference.id).kind != found->target_kind)
                return complete(CUDNN_STATUS_BAD_PARAM);
        const auto emitted = count(*found, header.count, has_array);
        bool duplicate{};
        for (std::size_t i = 0; i != command.values.references.size(); ++i)
            for (std::size_t j = 0; j != i; ++j)
                duplicate |= command.values.references[i].id == command.values.references[j].id;
        if (has_array) {
            for (std::int64_t i = 0; i < emitted; ++i) {
                auto& target = nodes.at(command.values.references[static_cast<std::size_t>(i)].id);
                if (header.name == 202 && target.finalized) {
                    ++duplicate_errors;
                    return complete(CUDNN_STATUS_BAD_PARAM);
                }
                target.finalized = true;
                target.value = 1000 + i;
            }
        }
        if (duplicate) {
            ++duplicate_successes;
            expect(header.name != 202 && nodes.at(command.values.references[0].id).value == 1002,
                   "successful ordered duplicates retain last native write");
        }
        for (std::size_t i = static_cast<std::size_t>(std::max<std::int64_t>(emitted, 0));
             has_array && i < command.values.references.size(); ++i)
            expect(nodes.at(command.values.references[i].id).value == kMarker,
                   "unemitted prepared tail is not a generated mutation");
        ExecutionPayloadWriter output;
        expect(write_execution_dnn_backend_get_result(
                   output, {has_count ? 8U : 0U,
                            has_count ? static_cast<std::uint64_t>(emitted) : 0U,
                            {15, {}, {}}}),
               "Get receipt contains only original count, never server descriptor pointers");
        auto result = complete();
        result.data = output.take();
        return result;
    }

    ExecutionWireReply handle(const ExecutionReceivedRequest& request) {
        const auto opcode = request.message.header.operation;
        ExecutionPayloadReader input(request.message.payload);
        if (opcode == ExecutionOpcode::kHello) {
            auto role = ExecutionBootstrapOwnership::kRetained;
            if (decode_execution_context_bootstrap(request.message.payload))
                execution = request.connection;
            else {
                std::uint32_t kind{}, flags{};
                expect(input.read_u32(kind) && kind == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty(),
                       "DNM1 bootstrap");
                metadata = request.connection;
                role = ExecutionBootstrapOwnership::kMetadataRetained;
            }
            const auto encoded = encode_execution_bootstrap_ownership(role);
            expect(encoded.has_value(), "bootstrap role encoding");
            auto result = complete(std::nullopt);
            result.data.assign(encoded->begin(), encoded->end());
            return result;
        }
        if (opcode == ExecutionOpcode::kDnnBackendCreate) {
            ExecutionDnnBackendCreate value;
            expect(request.connection == metadata &&
                       read_execution_dnn_backend_create(input, value) && input.empty() &&
                       value.has_output,
                   "all existing and generated descriptor identities remain DNM1");
            const auto id = next_id++;
            nodes.emplace(id, Node{value.kind});
            auto result = complete();
            result.id = id;
            return result;
        }
        if (opcode == ExecutionOpcode::kDnnBackendContext) {
            ExecutionDnnBackendContext envelope;
            expect(request.connection == execution &&
                       read_execution_dnn_backend_context(input, envelope) && input.empty() &&
                       envelope.metadata == metadata && !envelope.handle,
                   "planning wrappers authenticate context while retaining metadata owner");
            ExecutionPayloadReader nested(envelope.payload);
            if (envelope.operation == ExecutionOpcode::kDnnBackendFinalize) {
                std::uint64_t id{};
                expect(nested.read_u64(id) && nested.empty() &&
                           (nodes.at(id).kind == 2 || nodes.at(id).kind == 3 ||
                            nodes.at(id).kind == 4),
                       "all three source Finalizes use execution channel");
                nodes.at(id).finalized = true;
                ++finalizes;
                return complete();
            }
            return attribute(envelope.operation, nested);
        }
        if (opcode == ExecutionOpcode::kClose) {
            expect(request.connection == execution && input.empty(), "close owned context");
            return complete(std::nullopt);
        }
        std::uint64_t id{};
        expect(request.connection == metadata && input.read_u64(id) && input.empty(),
               "lifecycle remains context-independent after execution close");
        if (opcode == ExecutionOpcode::kDnnBackendInitialize) {
            auto& value = nodes.at(id);
            value.finalized = false;
            value.value = kMarker;
        } else {
            expect(opcode == ExecutionOpcode::kDnnBackendDestroy && nodes.erase(id) == 1,
                   "destroy one exact existing owner");
        }
        return complete();
    }
};

void run(const char* provider) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "current process authentication");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-heuristic-front-" + std::to_string(::getpid()) + ".sock";
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(!::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) &&
               !::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) &&
               !::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1),
           "configure synthetic local service");
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "actual authenticated socket transport");
    auto client = std::async(std::launch::async, [provider] { application(provider); });
    Peer peer;
    try {
        glimmer::tests::ExecutionDnnProfileFixture profiles;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(std::chrono::steady_clock::now() < deadline && server.poll_for(1),
                   "bounded frontend transport progress");
            if (auto request = server.take_request()) {
                expect(request->peer == *identity, "every request has the same authenticated peer");
                if (profiles.respond(server, *request)) continue;
                const auto reply = encode_execution_reply(peer.handle(*request));
                expect(reply && decode_execution_reply(*reply),
                       "structurally valid synthetic receipt");
                auto header = request->message.header;
                header.kind = ExecutionMessageKind::kReply;
                header.session = request->connection;
                header.incarnation = {79, 101};
                expect(server.reply(request->connection, {header, *reply}),
                       "reply to original connection");
            }
        }
        client.get();
        profiles.verify();
        expect(peer.nodes.empty() && peer.setters == 2 && peer.finalizes == 3 &&
                   peer.duplicate_errors == 1 && peer.duplicate_successes == 2,
               "every owner cleaned and original alias behavior distinguished");
        for (std::size_t i = 0; i != kFamilies.size(); ++i)
            expect(peer.preparations[i] == expected_requests(i).size() &&
                       peer.preparations[i] == peer.gets[i] + (i == 2 ? 2U : 1U),
                   "stale destination and bounded oversized capacity stop before final request");
    } catch (...) {
        server.stop();
        if (client.valid()) {
            expect(client.wait_for(std::chrono::seconds{5}) == std::future_status::ready,
                   "client terminates after endpoint failure");
            try {
                client.get();
            } catch (...) {
            }
        }
        throw;
    }
    server.stop();
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2, "synthetic graph provider path");
        run(argv[1]);
        std::cout << "Generated planning frontend transport tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
