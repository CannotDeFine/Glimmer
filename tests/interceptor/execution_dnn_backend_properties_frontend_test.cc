#include "execution_dnn_profile_fixture.h"
#include "glimmer/control/execution_dnn_backend_protocol.h"
#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_dnn_backend_graph_native_calls();
extern "C" unsigned int glimmer_dnn_native_calls();

namespace {
using namespace glimmer::control;
using Native = cudnnBackendDescriptor_t;
using Direction = ExecutionDnnBackendAttributeDirection;
using Operation = ExecutionDnnBackendAttributeOperation;
constexpr unsigned kCases = 20;
constexpr std::int64_t kMarker = -719;
constexpr std::array<std::uint32_t, 3> kNotes{0, 2, 3};
// Synthetic byte-contract oracle, not a claim about the vendor JSON schema.
const std::string kJson = "{\"fixture\":\"" + std::string(600, 'x') + "\"}";

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(cudnnStatus_t status) {
    expect(status == CUDNN_STATUS_SUCCESS, "owned synthetic original success");
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
    expect(!::dlclose(library), "cached routes survive provider close");
    return api;
}
Native create(const Api& api, unsigned kind) {
    Native value{};
    success(api.create(static_cast<cudnnBackendDescriptorType_t>(kind), &value));
    expect(value, "nonzero existing DNM1 token");
    return value;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): public attribute ABI helpers.
cudnnStatus_t set(const Api& api, Native value, unsigned name, unsigned type, std::int64_t count,
                  const void* array) {
    return api.set(value, static_cast<cudnnBackendAttributeName_t>(name),
                   static_cast<cudnnBackendAttributeType_t>(type), count, array);
}
cudnnStatus_t get(const Api& api, Native value, unsigned name, unsigned type,
                  std::int64_t requested, std::int64_t* count, void* array) {
    return api.get(value, static_cast<cudnnBackendAttributeName_t>(name),
                   static_cast<cudnnBackendAttributeType_t>(type), requested, count, array);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
class Pages {
   public:
    Pages() {
        const auto size = ::sysconf(_SC_PAGESIZE);
        expect(size > 0, "page size");
        page_ = static_cast<std::size_t>(size);
        memory_ = ::mmap(nullptr, page_ * 2, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(memory_ != MAP_FAILED, "guarded exact byte prefix");
        expect(!::mprotect(memory_, page_, PROT_READ | PROT_WRITE), "writable first page");
    }
    ~Pages() {
        if (::munmap(memory_, page_ * 2)) std::terminate();
    }
    Pages(const Pages&) = delete;
    Pages& operator=(const Pages&) = delete;
    void* tail(std::size_t bytes) const {
        expect(bytes <= page_, "bounded guarded prefix");
        return static_cast<std::byte*>(memory_) + page_ - bytes;
    }
    void read_only() const {
        expect(!::mprotect(memory_, page_, PROT_READ), "caller token array is read-only");
    }

   private:
    void* memory_{};
    std::size_t page_{};
};

void ordinary_application(const Api& api) {
    std::array<Native, 3> parents{create(api, 2), create(api, 4), create(api, 5)};
    const auto property = create(api, 38), json = create(api, 38), discovery = create(api, 38);
    expect(api.finalize(discovery) == CUDNN_STATUS_NOT_INITIALIZED,
           "default property requests a frame, but no current execution owner exists");
    cudnnHandle_t missing{};
    expect(set(api, property, 2701, 0, 1, &missing) == CUDNN_STATUS_NOT_INITIALIZED,
           "handle Set requires execution authority even for a safe null operand");
    CUcontext context{};
    expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS, "create actual owned synthetic context");
    cudnnHandle_t handle{};
    success(cudnnCreate(&handle));
    success(set(api, property, 2701, 0, 1, &handle));
    success(api.finalize(discovery));
    success(api.initialize(discovery));
    success(api.finalize(discovery));
    for (auto parent : parents) success(api.finalize(parent));
    Pages input;
    std::memcpy(input.tail(kJson.size()), kJson.data(), kJson.size());
    input.read_only();
    success(set(api, json, 2702, 24, static_cast<std::int64_t>(kJson.size()),
                input.tail(kJson.size())));
    success(cudnnDestroy(handle));
    expect(set(api, discovery, 2701, 0, 1, &handle) == CUDNN_STATUS_NOT_INITIALIZED,
           "stale handle rejected before final Set without renewing historical authority");
    expect(cuCtxDestroy(context) == CUDA_SUCCESS, "origin context closes before CPU metadata use");
    success(api.finalize(property));
    success(api.finalize(json));
    success(api.initialize(property));
    expect(api.finalize(property) == CUDNN_STATUS_NOT_INITIALIZED,
           "Initialize discards old populated-mode proof and needs discovery again");

    const auto copied = create(api, 38);
    for (std::size_t i = 0; i != parents.size(); ++i) {
        const auto name = std::array<unsigned, 3>{1307, 204, 407}[i];
        auto count = kMarker;
        const auto token = copied;
        success(get(api, parents[i], name, 15, 1, &count, const_cast<Native*>(&token)));
        expect(count == 1 && token == copied, "copied property mutates owner, never caller token");
        success(api.finalize(copied));
    }
    const auto layout_a = create(api, 9), layout_b = create(api, 9), unused = create(api, 9);
    {
        const std::array targets{layout_a, layout_b, layout_a, unused};
        Pages tokens;
        std::memcpy(tokens.tail(sizeof(targets)), targets.data(), sizeof(targets));
        tokens.read_only();
        auto count = kMarker;
        success(get(api, parents[0], 1304, 15, 4, &count, tokens.tail(sizeof(targets))));
        expect(count == 3 &&
                   !std::memcmp(tokens.tail(sizeof(targets)), targets.data(), sizeof(targets)),
               "full requested token prefix preserves ordered aliases and unwritten tail");
    }
    constexpr std::array<std::array<unsigned, 3>, 3> observations{
        {{0, 1303, 11}, {0, 1305, 19}, {1, 1101, 12}}};
    for (const auto& tuple : observations) {
        auto source = tuple[0] ? layout_a : parents[0];
        for (const auto requested : {INT64_MIN, std::int64_t{-1}, std::int64_t{0}, std::int64_t{1},
                                     std::int64_t{3}, INT64_MAX}) {
            const auto expected = requested < 0 ? 0U
                                  : requested == 0
                                      ? 3U
                                      : static_cast<unsigned>(std::min<std::int64_t>(requested, 3));
            auto count = kMarker;
            success(get(api, source, tuple[1], tuple[2], requested, &count, nullptr));
            expect(count == expected, "native enum count-only preserves signed and huge requests");
            Pages bytes;
            if (requested < 0) {
                expect(get(api, source, tuple[1], tuple[2], requested, &count, bytes.tail(0)) ==
                           CUDNN_STATUS_BAD_PARAM,
                       "negative array capacity never accesses PROT_NONE");
                continue;
            }
            const auto output_bytes = static_cast<std::size_t>(expected) * sizeof(std::uint32_t);
            success(
                get(api, source, tuple[1], tuple[2], requested, nullptr, bytes.tail(output_bytes)));
            expect(!std::memcmp(bytes.tail(output_bytes), kNotes.data(), output_bytes),
                   "request zero writes exact all-values prefix with no tail access");
        }
    }
    for (auto parent : parents) success(api.destroy(parent));
    std::int64_t count = kMarker, uid = kMarker;
    success(get(api, layout_a, 1100, 3, 1, &count, &uid));
    expect(count == 1 && uid == 1002, "last written alias survives source/context destruction");
    success(get(api, layout_b, 1100, 3, 1, &count, &uid));
    expect(uid == 1001, "independent generated target retains its own copied value");
    expect(get(api, unused, 1100, 3, 1, &count, &uid) == CUDNN_STATUS_NOT_INITIALIZED,
           "unwritten target is not invented as finalized");
    success(api.initialize(layout_a));
    expect(get(api, layout_a, 1100, 3, 1, &count, &uid) == CUDNN_STATUS_NOT_INITIALIZED,
           "Layout reset does not invent readable zero defaults");

    Pages output;
    count = kMarker;
    success(get(api, json, 2702, 24, 0, &count, nullptr));
    expect(count == static_cast<std::int64_t>(kJson.size()),
           "JSON count reports native byte length");
    std::memset(output.tail(kJson.size()), 0x55, kJson.size());
    for (const auto requested : {static_cast<std::int64_t>(kJson.size()), INT64_MAX}) {
        success(get(api, json, 2702, 24, requested, &count, output.tail(kJson.size())));
        expect(!std::memcmp(output.tail(kJson.size()), kJson.data(), kJson.size()),
               "JSON writes exact bytes beyond old 256-byte scratch, without a trailing NUL");
    }
    count = kMarker;
    expect(get(api, json, 2702, 24, static_cast<std::int64_t>(kJson.size() - 1), &count,
               output.tail(0)) == CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT &&
               count == kMarker,
           "short positive JSON preserves original 2006 without accessing either output");
    expect(get(api, json, 2702, 24, static_cast<std::int64_t>(kJson.size()), nullptr,
               output.tail(kJson.size())) == CUDNN_STATUS_BAD_PARAM,
           "unsafe successful null-count form is rejected before requested native call");
    expect(get(api, json, 2702, 24, INT64_MAX, &count, nullptr) == CUDNN_STATUS_BAD_PARAM,
           "positive null-array JSON is not silently turned into count-only");
    success(get(api, copied, 2700, 23, INT64_MIN, &count, nullptr));
    expect(count == 1, "copied property remains queryable after all source parents are destroyed");
    for (auto value : {property, json, discovery, copied, layout_a, layout_b, unused})
        success(api.destroy(value));
}

ExecutionWireReply complete(std::optional<std::int32_t> status = CUDNN_STATUS_SUCCESS) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = status;
    reply.status = status.value_or(0);
    return reply;
}
ExecutionWireReply rejected(CUresult status) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kRejected;
    reply.status = status;
    return reply;
}
struct Node {
    unsigned kind{};
    std::uint64_t generation{1};
    bool populated{}, finalized{};
    std::int64_t uid{};
    std::string json{};
};
struct Peer {
    unsigned test{}, direct_finalizes{}, framed_finalizes{}, handle_sets{}, copies{}, resets{};
    unsigned note_prepares{}, note_huge{}, note_count_only{}, note_without_count{}, json_huge{};
    std::uint64_t metadata{}, execution{}, next{100};
    bool handle_live{}, context_live{};
    std::unordered_map<std::uint64_t, Node> nodes;
    std::unordered_map<std::uint64_t, ExecutionDnnBackendAttributeHeader> pending;

    ExecutionWireReply hostile(bool framed) {
        if (!framed) {
            if (test == 1) return complete(CUDNN_STATUS_BAD_PARAM);
            if (test == 2) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            auto reply = rejected(CUDA_ERROR_INVALID_CONTEXT);
            if (test == 3) reply.parameter_bytes = 1;
            if (test == 4) reply.id = 1;
            if (test == 5) reply.address = 1;
            if (test == 6) reply.data = {std::byte{1}};
            if (test == 7) {
                reply.parameter_bytes = 1;
                reply.arguments = {{0, 1}};
            }
            if (test == 8) {
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.parameter_bytes = 1;
            }
            if (test == 9) return complete(CUDNN_STATUS_NOT_INITIALIZED);
            if (test == 10) {
                reply = complete();
                reply.parameter_bytes = 1;
            }
            if (test == 15) {
                reply = complete();
                reply.data = {std::byte{1}};
            }
            if (test == 16) {
                reply.disposition = ExecutionWireDisposition::kComputeReplacement;
                reply.status = 0;
                reply.id = 1;
            }
            if (test == 17) reply.status = CUDA_ERROR_INVALID_VALUE;
            if (test == 18) return complete();
            return reply;
        }
        if (test == 11) return rejected(CUDA_ERROR_INVALID_CONTEXT);
        if (test == 12) return complete(CUDNN_STATUS_BAD_PARAM);
        if (test == 13) {
            auto reply = rejected(CUDA_ERROR_INVALID_CONTEXT);
            reply.original_status = CUDNN_STATUS_SUCCESS;
            reply.parameter_bytes = 2;
            return reply;
        }
        auto reply = complete();
        reply.id = 1;  // case14: malformed second reply may not trigger a third call.
        return reply;
    }

    ExecutionWireReply lifecycle(ExecutionOpcode opcode, ExecutionPayloadReader& input,
                                 bool framed) {
        std::uint64_t id{};
        expect(input.read_u64(id) && input.empty(), "exact lifecycle identity");
        auto& node = nodes.at(id);
        if (opcode == ExecutionOpcode::kDnnBackendFinalize) {
            framed ? ++framed_finalizes : ++direct_finalizes;
            if (test) return hostile(framed);
            if (node.kind == 38 && !node.populated && !framed)
                return rejected(CUDA_ERROR_INVALID_CONTEXT);
            expect(node.kind == 38 ? !node.populated == framed : framed,
                   "only default/ID properties retry via execution, copied CPU modes stay DNM1");
            node.populated = true;
            node.finalized = true;
            ++node.generation;
        } else if (opcode == ExecutionOpcode::kDnnBackendInitialize) {
            expect(!framed, "native Initialize is context-independent");
            node.finalized = node.populated = false;
            node.json.clear();
            ++node.generation;
            ++resets;
        } else {
            expect(opcode == ExecutionOpcode::kDnnBackendDestroy && !framed,
                   "native descriptor destruction stays on original DNM1 owner");
            nodes.erase(id);
        }
        return complete();
    }

    ExecutionWireReply attribute(ExecutionOpcode opcode, ExecutionPayloadReader& input, bool framed,
                                 std::uint64_t handle) {
        ExecutionDnnBackendAttribute command;
        ExecutionDnnBackendAttributeHeader header;
        Operation operation{};
        const bool preparing = opcode == ExecutionOpcode::kDnnBackendPrepare;
        if (preparing) {
            ExecutionDnnBackendPrepare value;
            expect(read_execution_dnn_backend_prepare(input, value) && input.empty(),
                   "exact attribute preparation wire");
            header = value.header;
            operation = value.operation;
        } else {
            expect(read_execution_dnn_backend_attribute(input, command) && input.empty(),
                   "exact final attribute wire");
            header = command.header;
            operation =
                opcode == ExecutionOpcode::kDnnBackendSet ? Operation::kSet : Operation::kGet;
        }
        auto& node = nodes.at(header.descriptor);
        if (preparing) {
            pending.insert_or_assign(header.descriptor, header);
            if (header.name == 1303 || header.name == 1305 || header.name == 1101) {
                ++note_prepares;
                note_huge += header.count == INT64_MAX;
                note_count_only += header.presence == 2;
                note_without_count += header.presence == 1;
            }
            if (header.name == 2702 && operation == Operation::kGet)
                json_huge += header.count == INT64_MAX;
        } else {
            const auto& original = pending.at(header.descriptor);
            expect(header.name == original.name && header.type == original.type &&
                       header.count == original.count && header.presence == original.presence,
                   "final request preserves every prepared raw count/type/presence field");
            pending.erase(header.descriptor);
        }
        const bool has_array = (header.presence & 1) != 0, has_count = (header.presence & 2) != 0;
        const bool is_handle = operation == Operation::kSet && header.name == 2701;
        expect(framed == is_handle && (!framed || context_live),
               "only property HANDLE Set uses a frame; audited CPU observations never require it");
        expect(!handle || (is_handle && !preparing && handle_live && handle == 21),
               "frame contains only a live translated logical handle");
        if (!preparing)
            expect(command.generation == node.generation, "final generation is unchanged");
        ExecutionDnnBackendAttributePreparation prepared{
            node.generation, Direction::kScalarOutput, 0, 0, has_count ? 8U : 0U, 0};
        const bool descriptor_output = header.type == 15;
        std::int64_t count{1};
        if (operation == Operation::kSet) {
            expect(node.kind == 38 && (is_handle || header.name == 2702), "known property setter");
            prepared.direction = Direction::kInput;
            prepared.count_bytes = 0;
            prepared.input_elements = static_cast<std::uint32_t>(header.count);
        } else if (descriptor_output) {
            prepared.direction = Direction::kDescriptorOutput;
            prepared.input_elements =
                has_array && header.count > 0 ? static_cast<std::uint32_t>(header.count) : 0;
            count = header.name == 1304 ? std::clamp<std::int64_t>(header.count, 0, 3) : 1;
        } else if (header.name == 2702) {
            expect(header.type == 24, "JSON uses exact CHAR storage");
            count = static_cast<std::int64_t>(node.json.size());
            if (header.count > 0 && header.count < count) {
                // Short positive output is a safe original error even for inaccessible spans.
            } else if (!has_count || (header.count > 0 && !has_array))
                return rejected(CUDA_ERROR_INVALID_VALUE);
            else if (header.count > 0 && has_array)
                prepared.output_elements = prepared.output_bytes =
                    static_cast<std::uint32_t>(count);
        } else if (header.name == 1303 || header.name == 1305 || header.name == 1101) {
            expect(header.type == (header.name == 1303   ? 11U
                                   : header.name == 1305 ? 19U
                                                         : 12U),
                   "exact raw enum attribute type");
            count = header.count == 0 ? 3 : std::clamp<std::int64_t>(header.count, 0, 3);
            if (has_array && header.count >= 0) {
                prepared.output_elements = static_cast<std::uint32_t>(count);
                prepared.output_bytes = prepared.output_elements * 4;
            }
        } else {
            expect(header.name == 1100 || header.name == 2700, "known scalar getter");
            if (has_array && header.count == 1) {
                prepared.output_elements = 1;
                prepared.output_bytes = header.name == 1100 ? 8 : 4;
            }
        }
        if (preparing) {
            ExecutionPayloadWriter bytes;
            expect(write_execution_dnn_backend_preparation(bytes, prepared),
                   "canonical preparation");
            auto reply = complete(std::nullopt);
            reply.data = bytes.take();
            return reply;
        }
        if (operation == Operation::kSet) {
            if (is_handle) {
                expect(
                    handle == 21 && command.values.references.size() == 1 &&
                        command.values.references[0].kind ==
                            ExecutionDnnBackendReferenceKind::kHandle &&
                        command.values.references[0].owner == execution &&
                        command.values.references[0].id == 21,
                    "handle capture contains authenticated logical identity, never caller address");
                ++handle_sets;
            } else {
                expect(command.values.scalars.size() == kJson.size() &&
                           !std::memcmp(command.values.scalars.data(), kJson.data(), kJson.size()),
                       "Set captures exact read-only JSON prefix including its final byte");
                node.json = kJson;
            }
            node.populated = true;
            ++node.generation;
            return complete();
        }
        if (!node.finalized) return complete(CUDNN_STATUS_NOT_INITIALIZED);
        if ((!has_array && !has_count) || (has_array && header.count < 0))
            return complete(CUDNN_STATUS_BAD_PARAM);
        if (header.name == 2702 && header.count > 0 && header.count < count)
            return complete(CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT);
        ExecutionDnnBackendGetResult result{has_count ? 8U : 0U,
                                            has_count ? static_cast<std::uint64_t>(count) : 0,
                                            {header.type, {}, {}}};
        if (descriptor_output) {
            expect(command.values.references.size() == prepared.input_elements,
                   "all requested destination identities captured, including unwritten tail");
            for (const auto& reference : command.values.references)
                expect(reference.kind == ExecutionDnnBackendReferenceKind::kDescriptor &&
                           reference.owner == metadata &&
                           nodes.at(reference.id).kind == (header.name == 1304 ? 9U : 38U),
                       "destination kinds and DNM1 ownership preserved");
            if (has_array)
                for (std::int64_t i = 0; i < count; ++i) {
                    auto& destination =
                        nodes.at(command.values.references[static_cast<std::size_t>(i)].id);
                    destination.finalized = destination.populated = true;
                    destination.uid = 1000 + i;
                    ++destination.generation;
                    ++copies;
                }
        } else {
            result.values.scalars.resize(prepared.output_bytes);
            if (prepared.output_bytes) {
                const void* data = header.name == 2702 ? static_cast<const void*>(node.json.data())
                                   : header.name == 1100 ? static_cast<const void*>(&node.uid)
                                                         : static_cast<const void*>(kNotes.data());
                std::memcpy(result.values.scalars.data(), data, prepared.output_bytes);
            }
        }
        ExecutionPayloadWriter bytes;
        expect(write_execution_dnn_backend_get_result(bytes, result),
               "exact successful output prefix");
        auto reply = complete();
        reply.data = bytes.take();
        return reply;
    }

    ExecutionWireReply handle(const ExecutionReceivedRequest& request) {
        const auto opcode = request.message.header.operation;
        ExecutionPayloadReader input(request.message.payload);
        if (opcode == ExecutionOpcode::kHello) {
            auto ownership = ExecutionBootstrapOwnership::kRetained;
            if (decode_execution_context_bootstrap(request.message.payload)) {
                execution = request.connection;
                context_live = true;
            } else {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty(),
                       "exact DNM1 bootstrap");
                metadata = request.connection;
                ownership = ExecutionBootstrapOwnership::kMetadataRetained;
            }
            auto reply = complete(std::nullopt);
            const auto encoded = encode_execution_bootstrap_ownership(ownership);
            expect(encoded.has_value(), "bootstrap ownership encoding");
            reply.data.assign(encoded->begin(), encoded->end());
            return reply;
        }
        if (opcode == ExecutionOpcode::kDnnBackendCreate) {
            ExecutionDnnBackendCreate value;
            expect(request.connection == metadata &&
                       read_execution_dnn_backend_create(input, value) && input.empty() &&
                       value.has_output,
                   "descriptor identities remain on DNM1");
            auto reply = complete();
            reply.id = next++;
            nodes.emplace(reply.id, Node{value.kind});
            return reply;
        }
        if (opcode == ExecutionOpcode::kDnnBackendContext) {
            ExecutionDnnBackendContext value;
            expect(request.connection == execution && context_live &&
                       read_execution_dnn_backend_context(input, value) && input.empty() &&
                       value.metadata == metadata,
                   "authenticated current execution envelope");
            ExecutionPayloadReader nested(value.payload);
            if (value.operation == ExecutionOpcode::kDnnBackendFinalize) {
                expect(!value.handle, "discovery does not fabricate a handle");
                return lifecycle(value.operation, nested, true);
            }
            return attribute(value.operation, nested, true, value.handle);
        }
        if (opcode == ExecutionOpcode::kCreateDnn) {
            expect(request.connection == execution && input.empty() && !handle_live,
                   "create owned handle");
            handle_live = true;
            auto reply = complete();
            reply.id = 21;
            return reply;
        }
        if (opcode == ExecutionOpcode::kDestroyDnn) {
            std::uint64_t id{};
            expect(request.connection == execution && input.read_u64(id) && id == 21 &&
                       input.empty() && handle_live,
                   "destroy source handle without invalidating copied CPU data");
            handle_live = false;
            return complete();
        }
        if (opcode == ExecutionOpcode::kClose) {
            expect(request.connection == execution && input.empty() && !handle_live,
                   "context-first close");
            context_live = false;
            return complete(std::nullopt);
        }
        expect(request.connection == metadata, "CPU operations preserve original metadata owner");
        if (opcode == ExecutionOpcode::kDnnBackendPrepare ||
            opcode == ExecutionOpcode::kDnnBackendSet || opcode == ExecutionOpcode::kDnnBackendGet)
            return attribute(opcode, input, false, 0);
        return lifecycle(opcode, input, false);
    }
};

cudnnStatus_t hostile_status(unsigned test) {
    if (test == 1 || test == 12 || test == 17) return CUDNN_STATUS_BAD_PARAM;
    if (test == 2) return CUDNN_STATUS_NOT_SUPPORTED;
    if (test == 3 || test == 9 || test == 11 || test == 13) return CUDNN_STATUS_NOT_INITIALIZED;
    if (test == 18) return CUDNN_STATUS_SUCCESS;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
void run(const char* provider, unsigned test) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "current authenticated peer");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-properties-front-" + std::to_string(::getpid()) + ".sock";
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(!::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) &&
               !::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) &&
               !::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1),
           "local fake endpoint");
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "actual authenticated IPC transport");
    auto client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        if (!test)
            ordinary_application(api);
        else {
            const auto property = create(api, 38);
            CUcontext context{};
            expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS,
                   "current owner exists for retry controls");
            expect(api.finalize(property) == hostile_status(test),
                   "strict finalize receipt/status containment");
            // Poison controls deliberately retain their uncertain owners. Healthy
            // controls clean up normally and prove no extra Finalize was needed.
            if (test == 1 || test == 2 || test == 9 || test == 11 || test == 12 || test == 17 ||
                test == 18) {
                success(api.destroy(property));
                expect(cuCtxDestroy(context) == CUDA_SUCCESS, "healthy control cleanup");
            }
        }
        expect(!glimmer_dnn_backend_graph_native_calls() && !glimmer_dnn_native_calls(),
               "no original SDK consumer or fallback executes in the application");
    });
    Peer peer;
    peer.test = test;
    try {
        glimmer::tests::ExecutionDnnProfileFixture profiles;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(std::chrono::steady_clock::now() < deadline && server.poll_for(1),
                   "bounded transport progress");
            if (auto request = server.take_request()) {
                expect(request->peer == *identity, "same actual peer on both connections");
                if (profiles.respond(server, *request)) continue;
                const auto reply = peer.handle(*request);
                if (test == 19 &&
                    request->message.header.operation == ExecutionOpcode::kDnnBackendFinalize) {
                    server.close_peer(request->connection);
                    continue;  // Ambiguous first reply must never reach execution retry.
                }
                const auto encoded = encode_execution_reply(reply);
                expect(encoded && decode_execution_reply(*encoded),
                       "wire-valid hostile receipt reaches semantic guard");
                auto header = request->message.header;
                header.kind = ExecutionMessageKind::kReply;
                header.session = request->connection;
                header.incarnation = {113, 127};
                expect(server.reply(request->connection, {header, *encoded}),
                       "reply to exact original owner");
            }
        }
        client.get();
        profiles.verify();
        if (!test)
            expect(
                peer.nodes.empty() && !peer.context_live && !peer.handle_live &&
                    peer.direct_finalizes == 9 && peer.framed_finalizes == 5 &&
                    peer.handle_sets == 1 && peer.copies == 6 && peer.resets == 3 &&
                    peer.note_prepares == 36 && peer.note_huge == 6 && peer.note_count_only == 18 &&
                    peer.note_without_count == 12 && peer.json_huge == 2,
                "all normal lifecycle/context transitions and copied prefixes accounted exactly");
        else
            expect(peer.direct_finalizes == 1 &&
                       peer.framed_finalizes == (test >= 11 && test <= 14 ? 1U : 0U),
                   "only one exact clean preissue response retries; never a third Finalize");
    } catch (...) {
        server.stop();
        if (client.valid()) {
            expect(client.wait_for(std::chrono::seconds{5}) == std::future_status::ready,
                   "client unblocks after endpoint failure");
            try {
                client.get();
            } catch (...) {
            }
        }
        throw;
    }
    server.stop();
}
unsigned number(const char* text) {
    unsigned value{};
    const std::string_view input{text};
    expect(!input.empty() && input.size() <= 2, "bounded isolated case");
    for (auto digit : input) {
        expect(digit >= '0' && digit <= '9', "decimal test selector");
        value = value * 10 + static_cast<unsigned>(digit - '0');
    }
    expect(value < kCases, "known isolated case");
    return value;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 4, "graph provider path and optional isolated case");
        if (argc == 4) {
            expect(std::string_view{argv[2]} == "--case", "isolated child argument");
            run(argv[1], number(argv[3]));
        } else {
            for (unsigned test = 0; test < kCases; ++test) {
                const auto text = std::to_string(test);
                const auto child = ::fork();
                expect(child >= 0, "fork isolated interceptor ownership");
                if (!child) {
                    ::execl("/proc/self/exe", argv[0], argv[1], "--case", text.c_str(),
                            static_cast<char*>(nullptr));
                    ::_exit(127);
                }
                int status{};
                pid_t waited{};
                do {
                    waited = ::waitpid(child, &status, 0);
                } while (waited < 0 && errno == EINTR);
                if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
                    throw std::runtime_error("properties frontend case failed: " + text);
            }
        }
        std::cout << "Engine/LayoutInfo/property frontend transport checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
