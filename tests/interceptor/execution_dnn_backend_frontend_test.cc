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
using Clock = std::chrono::steady_clock;
using Native = cudnnBackendDescriptor_t;
using Class = ExecutionDnnBackendBindingClass;
using Role = ExecutionDnnBackendBindingRole;
using Access = ExecutionDnnBackendBindingAccess;
using Direction = ExecutionDnnBackendAttributeDirection;
using Operation = ExecutionDnnBackendAttributeOperation;
constexpr unsigned int kCases = 44;
constexpr std::uint64_t kAddress = 0x100000;
constexpr std::array<std::int64_t, 7> kUids{7, 8, 9, 10, 11, 12, 7};

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t status) {
    expect(status == CUDNN_STATUS_SUCCESS, "public synthetic original SDK success");
}
void driver(CUresult status) {
    expect(status == CUDA_SUCCESS, "public owned Driver transport success");
}
void* device_pointer(CUdeviceptr address) {
    // Synthetic CUDA address; the application never dereferences it.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(address);
}
struct Api {
    decltype(&cudnnBackendCreateDescriptor) create{};
    decltype(&cudnnBackendDestroyDescriptor) destroy{};
    decltype(&cudnnBackendInitialize) initialize{};
    decltype(&cudnnBackendFinalize) finalize{};
    decltype(&cudnnBackendSetAttribute) set{};
    decltype(&cudnnBackendGetAttribute) get{};
    decltype(&cudnnBackendExecute) execute{};
};
Api lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "open graph provider without loading native CUDA/cuDNN");
    Api api;
#define LOOKUP(member, symbol)                                                     \
    do {                                                                           \
        auto* value = ::dlsym(library, #symbol);                                   \
        expect(value == reinterpret_cast<void*>(&(symbol)) &&                      \
                   value == ::dlsym(RTLD_DEFAULT, #symbol),                        \
               "direct/global/explicit graph symbol routes to owned interceptor"); \
        api.member = reinterpret_cast<decltype(api.member)>(value);                \
    } while (false)
    LOOKUP(create, cudnnBackendCreateDescriptor);
    LOOKUP(destroy, cudnnBackendDestroyDescriptor);
    LOOKUP(initialize, cudnnBackendInitialize);
    LOOKUP(finalize, cudnnBackendFinalize);
    LOOKUP(set, cudnnBackendSetAttribute);
    LOOKUP(get, cudnnBackendGetAttribute);
    LOOKUP(execute, cudnnBackendExecute);
#undef LOOKUP
    expect(::dlclose(library) == 0, "cached owned routes survive provider close");
    return api;
}
cudnnStatus_t set(const Api& api, Native descriptor, unsigned name, unsigned type,
                  std::int64_t count, const void* array) {
    return api.set(descriptor, static_cast<cudnnBackendAttributeName_t>(name),
                   static_cast<cudnnBackendAttributeType_t>(type), count, array);
}
cudnnStatus_t get(const Api& api, Native descriptor, unsigned name, unsigned type,
                  std::int64_t* count, void* array) {
    return api.get(descriptor, static_cast<cudnnBackendAttributeName_t>(name),
                   static_cast<cudnnBackendAttributeType_t>(type), 1, count, array);
}
class Pages {
   public:
    Pages() {
        const auto result = ::sysconf(_SC_PAGESIZE);
        expect(result > 0, "page size is available");
        page = static_cast<std::size_t>(result);
        memory = ::mmap(nullptr, page * 2, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(memory != MAP_FAILED, "two inaccessible pages for exact capture proof");
    }
    ~Pages() {
        if (::munmap(memory, page * 2)) std::terminate();
    }
    Pages(const Pages&) = delete;
    Pages& operator=(const Pages&) = delete;
    void* tail(std::size_t bytes) const {
        return static_cast<std::byte*>(memory) + page - bytes;
    }
    void expose() const {
        expect(::mprotect(memory, page, PROT_READ | PROT_WRITE) == 0,
               "make only first page readable");
    }
    void seed(unsigned round) const {
        expose();
        const std::array<std::byte, 8> bytes{std::byte{0x10}, std::byte{0x21},
                                             std::byte{0x32}, std::byte{0x43},
                                             std::byte{0x54}, std::byte{0x65},
                                             std::byte{0x76}, static_cast<std::byte>(0x80 + round)};
        std::memcpy(tail(8), bytes.data(), bytes.size());
    }
    void* memory{};
    std::size_t page{};
};
struct Expected {
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    bool metadata{}, execution{}, has_final{};
};
Expected expected(unsigned test) {
    if (!test) return {CUDNN_STATUS_SUCCESS, true, true, true};
    if (test == 3 || test == 25) return {CUDNN_STATUS_INTERNAL_ERROR, false, true, test == 25};
    if (test == 4 || test == 23 || test == 26)
        return {CUDNN_STATUS_NOT_INITIALIZED, true, false, test != 4};
    if (test == 5) return {CUDNN_STATUS_NOT_SUPPORTED, true, true, false};
    if (test == 17) return {CUDNN_STATUS_BAD_PARAM, true, true, true};
    if (test == 43) return {CUDNN_STATUS_BAD_PARAM, true, true, true};
    if (test == 27 || test == 41) return {CUDNN_STATUS_NOT_SUPPORTED, true, true, true};
    if (test >= 35 && test <= 38) return {CUDNN_STATUS_BAD_PARAM, true, true, false};
    return {CUDNN_STATUS_INTERNAL_ERROR, false, false, (test >= 18 && test <= 34) || test == 42};
}

struct Application {
    Api api;
    CUcontext context{};
    std::array<cudnnHandle_t, 2> handles{};
    std::array<Native, 5> descriptors{};  // Graph, Engine, EngineConfig, Plan, VariantPack.
    CUdeviceptr memory{};
    Pages pages;
    explicit Application(const char* provider) : api(lookup(provider)) {
        constexpr std::array<unsigned, 5> kinds{15, 2, 3, 5, 16};
        for (std::size_t i = 0; i != kinds.size(); ++i) {
            check(api.create(static_cast<cudnnBackendDescriptorType_t>(kinds[i]), &descriptors[i]));
            expect(descriptors[i], "nonzero DNM1 descriptor token");
        }
        driver(cuCtxCreate(&context, 0, 0));
        for (auto& handle : handles) check(cudnnCreate(&handle));
        driver(cuMemAlloc(&memory, 64));
        expect(memory == kAddress, "synthetic allocation has stable owned numeric base");
        expect(set(api, descriptors[0], 800, 0, 2, handles.data()) == CUDNN_STATUS_BAD_PARAM,
               "two distinct logged handles reach original count error unchanged");
        check(set(api, descriptors[0], 800, 0, 1, handles.data()));
        check(set(api, descriptors[1], 1300, 15, 1, &descriptors[0]));
        check(set(api, descriptors[2], 300, 15, 1, &descriptors[1]));
        check(set(api, descriptors[3], 401, 15, 1, &descriptors[2]));
        check(set(api, descriptors[3], 400, 0, 1, handles.data()));
        for (const auto descriptor : std::span{descriptors}.first(4))
            check(api.finalize(descriptor));
        const std::array<void*, 7> pointers{device_pointer(memory), pages.tail(8), pages.tail(4),
                                            pages.tail(2),          pages.tail(0), nullptr,
                                            device_pointer(memory)};
        // All host pointees are PROT_NONE here; Set is allowed to read pointer values only.
        check(set(api, descriptors[4], 1001, 6, pointers.size(), pointers.data()));
        check(set(api, descriptors[4], 1000, 3, kUids.size(), kUids.data()));
        void* workspace = device_pointer(memory + 64);
        check(set(api, descriptors[4], 1003, 6, 1, &workspace));
        check(api.finalize(descriptors[4]));
        void* returned{};
        std::int64_t count = -1;
        check(get(api, descriptors[4], 1003, 6, &count, &returned));
        expect(returned == workspace && count == 1,
               "workspace Get reproduces caller slot identity without a server address");
    }
    cudnnStatus_t execute() const {
        return api.execute(handles[0], descriptors[3], descriptors[4]);
    }
    void close() {
        driver(cuMemFree(memory));
        for (const auto handle : handles) check(cudnnDestroy(handle));
        driver(cuCtxDestroy(context));
        cudnnHandle_t historical{};
        std::int64_t count = -1;
        check(get(api, descriptors[3], 400, 0, &count, &historical));
        expect(count == 1 && historical == handles[0],
               "historical handle output survives context teardown without renewed authority");
        check(api.initialize(descriptors[3]));
        for (const auto descriptor : descriptors) check(api.destroy(descriptor));
    }
};

ExecutionWireReply complete(std::optional<std::int32_t> original = CUDNN_STATUS_SUCCESS) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = original;
    reply.status = original.value_or(0);
    return reply;
}
ExecutionWireReply rejected(CUresult status, std::uint64_t faults) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kRejected;
    reply.status = status;
    reply.parameter_bytes = faults;
    return reply;
}

struct FakeService {
    unsigned test{};
    std::uint64_t metadata{}, execution{};
    unsigned creates{}, handle_creates{}, prepares{}, finals{}, metadata_probes{}, stream_probes{};
    unsigned context_prepares{}, context_finals{}, direct_finalizes{}, handle_set_errors{};
    std::vector<ExecutionDnnBackendReference> pointers;
    ExecutionDnnBackendReference workspace, plan_handle;
    std::unordered_map<std::uint64_t, unsigned> kinds;

    ExecutionWireReply attribute(ExecutionOpcode opcode, ExecutionPayloadReader& input,
                                 std::optional<std::uint64_t> frame_handle) {
        if (opcode == ExecutionOpcode::kDnnBackendPrepare) {
            ExecutionDnnBackendPrepare request;
            expect(read_execution_dnn_backend_prepare(input, request) && input.empty(),
                   "decode exact bounded attribute preparation");
            const auto kind = kinds.at(request.header.descriptor);
            expect(frame_handle.has_value() ==
                       (kind != 16 &&
                        (request.header.name != 400 || request.operation == Operation::kSet)),
                   "planning preparation uses execution channel except historical Get400");
            // Set400 remains context-owned; only Get400 is context-independent.
            if (request.header.name == 400)
                expect(frame_handle.has_value() == (request.operation == Operation::kSet),
                       "Plan HANDLE input/output have independent authority");
            if (frame_handle) {
                expect(!*frame_handle, "preparation has no guessed native handle");
                ++context_prepares;
            }
            ExecutionDnnBackendAttributePreparation prepared;
            prepared.generation = 19;
            if (request.operation == Operation::kSet) {
                prepared.direction = Direction::kInput;
                prepared.input_elements = static_cast<std::uint32_t>(request.header.count);
            } else {
                prepared.direction =
                    request.header.type == 0 ? Direction::kHandleOutput : Direction::kPointerOutput;
                prepared.output_elements = 1;
                prepared.count_bytes = request.header.presence & 2U ? 8 : 0;
            }
            ExecutionPayloadWriter output;
            expect(write_execution_dnn_backend_preparation(output, prepared),
                   "encode independently chosen direction and exact prefix");
            auto reply = complete(std::nullopt);
            reply.data = output.take();
            return reply;
        }
        ExecutionDnnBackendAttribute request;
        expect(read_execution_dnn_backend_attribute(input, request) && input.empty() &&
                   request.generation == 19,
               "final attribute carries exact preparation generation");
        const auto& header = request.header;
        const bool is_set = opcode == ExecutionOpcode::kDnnBackendSet;
        const auto kind = kinds.at(header.descriptor);
        expect(frame_handle.has_value() == (kind != 16 && (header.name != 400 || is_set)),
               "correct channel for final context-free or planning attribute");
        auto reply = complete();
        if (frame_handle) ++context_finals;
        if (is_set && header.type == 0) {
            expect(frame_handle == 21 && request.values.scalars.empty() &&
                       request.values.references.size() == static_cast<std::size_t>(header.count),
                   "HANDLE final frame binds first live handle and all logged elements");
            for (std::size_t i = 0; i != request.values.references.size(); ++i) {
                const auto& reference = request.values.references[i];
                expect(reference.kind == ExecutionDnnBackendReferenceKind::kHandle &&
                           reference.owner == execution && reference.id == 21 + i,
                       "logged HANDLE array keeps independent same-session IDs");
            }
            if (header.count == 2) {
                ++handle_set_errors;
                return complete(CUDNN_STATUS_BAD_PARAM);
            }
            if (header.name == 400) plan_handle = request.values.references[0];
        } else if (is_set && header.type == 15) {
            expect(frame_handle == 0 && request.values.references.size() == 1 &&
                       request.values.references[0].owner == metadata &&
                       request.values.references[0].kind ==
                           ExecutionDnnBackendReferenceKind::kDescriptor,
                   "copied child reference keeps original DNM1 authority, never migrates");
        } else if (is_set && header.type == 6) {
            expect(!frame_handle && request.values.scalars.empty(),
                   "pointer Set sends symbolic identities, never by-value bytes");
            for (const auto& reference : request.values.references)
                expect(reference.kind == ExecutionDnnBackendReferenceKind::kNull ||
                           (reference.kind == ExecutionDnnBackendReferenceKind::kPointerSlot &&
                            reference.owner == metadata && reference.id),
                       "each pointer binding is canonical null or bounded owner-local slot");
            if (header.name == 1001) {
                expect(request.values.references.size() == 7, "all ordered pointer slots retained");
                pointers = request.values.references;
                expect(pointers[0].id == pointers[6].id && !pointers[5].id,
                       "duplicate pointer identities and native null position preserved");
            } else {
                expect(header.name == 1003 && request.values.references.size() == 1,
                       "workspace has one symbolic binding");
                workspace = request.values.references[0];
            }
        } else if (is_set) {
            expect(header.name == 1000 && header.type == 3 && !frame_handle &&
                       request.values.references.empty() && request.values.scalars.size() == 56 &&
                       std::memcmp(request.values.scalars.data(), kUids.data(), 56) == 0,
                   "duplicate UID values retain exact original order");
        } else {
            ExecutionPayloadWriter output;
            const ExecutionDnnBackendGetResult result{
                header.presence & 2U ? 8U : 0U,
                header.presence & 2U ? 1U : 0U,
                {header.type, {}, {header.name == 400 ? plan_handle : workspace}}};
            expect(write_execution_dnn_backend_get_result(output, result),
                   "output identities contain no native pointer bits");
            reply.data = output.take();
        }
        return reply;
    }

    ExecutionDnnBackendExecutePreparation requirements() const {
        expect(pointers.size() == 7 && workspace.id, "Execute follows captured slot metadata");
        ExecutionDnnBackendExecutePreparation value{100 + prepares, 200 + prepares, {}};
        for (std::uint32_t index = 0; index != 7; ++index) {
            auto value_class = Class::kDevice;
            auto access = Access::kRead;
            std::uint64_t bytes = index ? 8 : 64, alignment = 1;
            if (index >= 1 && index <= 3) {
                value_class = Class::kByValue;
                bytes = alignment = 16U >> index;
            } else if (index == 4) {
                value_class = Class::kIgnored;
                access = Access::kNone;
                bytes = alignment = 0;
            } else if (index == 5) {
                value_class = Class::kByValue;
                bytes = UINT64_MAX;
                alignment = 8;
            }
            value.requirements.push_back({index, Role::kTensor, value_class, access, kUids[index],
                                          pointers[index].id, bytes, alignment});
        }
        value.requirements.push_back({UINT32_MAX, Role::kWorkspace, Class::kDevice,
                                      Access::kReadWrite, 0, workspace.id, 0, 1});
        return value;
    }

    ExecutionWireReply prepare_execute(ExecutionPayloadReader& input) {
        ExecutionDnnBackendExecutePrepare request;
        expect(read_execution_dnn_backend_execute_prepare(input, request) && input.empty() &&
                   request.handle == 21 && request.metadata == metadata && request.plan == 105 &&
                   request.variant == 116,
               "Execute preparation binds handle plus unchanged DNM1 plan/variant");
        ++prepares;
        auto prepared = requirements();
        if (prepares == 1) {
            if (test == 36) prepared.requirements[0].bytes = 65;
            if (test == 37) prepared.requirements[1].alignment = UINT64_MAX;
            if (test == 38) prepared.requirements[1].value_class = Class::kDevice;
            if (test == 43) prepared.plan_generation = 777;
        }
        ExecutionPayloadWriter output;
        expect(write_execution_dnn_backend_execute_preparation(output, prepared),
               "canonical service-derived requirements encode");
        auto reply = complete(std::nullopt);
        reply.data = output.take();
        if (prepares != 1) return reply;
        if (test == 1 || test == 2) reply.parameter_bytes = test;
        if (test == 3) return rejected(CUDA_ERROR_UNKNOWN, 1);
        if (test == 4) return rejected(CUDA_ERROR_INVALID_CONTEXT, 2);
        if (test == 5) return rejected(CUDA_ERROR_NOT_SUPPORTED, 0);
        if (test == 6) std::fill(reply.data.begin(), reply.data.begin() + 8, std::byte{0});
        if (test == 7) std::fill(reply.data.begin() + 8, reply.data.begin() + 16, std::byte{0});
        if (test == 8 || test == 39) {
            const auto offset = 20 + (test == 8 ? 0 : 4) * 48 + 24;
            std::fill(reply.data.begin() + offset, reply.data.begin() + offset + 8,
                      std::byte{0xff});
        }
        if (test == 9) std::fill(reply.data.begin() + 60, reply.data.begin() + 68, std::byte{0});
        if (test == 10) std::fill(reply.data.begin() + 68, reply.data.begin() + 72, std::byte{0});
        if (test == 11)
            std::fill(reply.data.begin() + 100, reply.data.begin() + 108, std::byte{0xff});
        if (test == 12) reply.original_status = CUDNN_STATUS_SUCCESS;
        if (test == 13) reply.data.push_back(std::byte{0});
        if (test == 14) reply.id = 71;
        if (test == 15) reply.address = 81;
        if (test == 16) reply.parameter_bytes = 4;
        if (test == 40) reply.data.clear();
        return reply;
    }

    ExecutionWireReply execute(ExecutionPayloadReader& input) {
        ExecutionDnnBackendExecute request;
        expect(
            read_execution_dnn_backend_execute(input, request) && input.empty() &&
                request.handle == 21 && request.metadata == metadata && request.plan == 105 &&
                request.variant == 116 &&
                request.plan_generation == (test == 43 && prepares == 1 ? 777U : 100 + prepares) &&
                request.variant_generation == 200 + prepares && request.bindings.size() == 8,
            "final admission echoes both current generations and complete ordered binding set");
        const auto prepared = requirements();
        for (std::size_t index = 0; index != request.bindings.size(); ++index) {
            const auto& binding = request.bindings[index];
            const auto& requirement = prepared.requirements[index];
            expect(binding.index == requirement.index &&
                       binding.value_class == requirement.value_class &&
                       binding.slot == requirement.slot,
                   "every final binding matches its original requirement index/class/slot");
            if (index == 0 || index == 6 || index == 7) {
                expect(binding.allocation == 41 && binding.offset == (index == 7 ? 64U : 0U) &&
                           binding.host_bytes.empty(),
                       "full-range and nonnull zero-byte endpoint retain owned device identity");
            } else if (index <= 3) {
                const std::array<std::byte, 8> all{
                    std::byte{0x10}, std::byte{0x21},
                    std::byte{0x32}, std::byte{0x43},
                    std::byte{0x54}, std::byte{0x65},
                    std::byte{0x76}, static_cast<std::byte>(0x80 + (prepares - 1))};
                const auto wanted = std::span{all}.last(16U >> index);
                expect(
                    !binding.allocation && !binding.offset &&
                        binding.host_bytes.size() == wanted.size() &&
                        std::equal(binding.host_bytes.begin(), binding.host_bytes.end(),
                                   wanted.begin()),
                    "by-value snapshots use exact 8/4/2 bytes changed after Set and per Execute");
            } else {
                expect(!binding.allocation && !binding.offset && binding.host_bytes.empty(),
                       "ignored inaccessible and active NULL never acquire storage");
            }
        }
        ++finals;
        auto reply = complete();
        reply.id = 500 + finals;
        if (prepares != 1) return reply;
        if (test == 43) return rejected(CUDA_ERROR_INVALID_VALUE, 0);
        if (test == 17) reply.status = *reply.original_status = CUDNN_STATUS_BAD_PARAM;
        if (test == 18) reply.id = 0;
        if (test == 19) reply.original_status.reset();
        if (test == 20) reply.status = CUDNN_STATUS_BAD_PARAM;
        if (test == 21 || test == 22) reply.parameter_bytes = test - 20;
        if (test == 23 || test == 24 || test == 42) {
            reply = rejected(CUDA_ERROR_INVALID_CONTEXT, test == 24 ? 1 : 2);
            reply.id = test == 42 ? 0 : 500;
            reply.original_status = CUDNN_STATUS_SUCCESS;
        }
        if (test == 25) reply = rejected(CUDA_ERROR_UNKNOWN, 1);
        if (test == 26) reply = rejected(CUDA_ERROR_INVALID_CONTEXT, 2);
        if (test == 27 || test == 41) {
            reply = rejected(CUDA_ERROR_NOT_SUPPORTED, 0);
            reply.id = test == 27 ? 500 : 0;
        }
        if (test == 28 || test == 29) {
            reply = rejected(CUDA_ERROR_UNKNOWN, 0);
            reply.status = test == 28 ? 0 : 1024;
        }
        if (test == 30) reply.status = *reply.original_status = 8192;
        if (test == 31) reply.data = {std::byte{1}};
        if (test == 32) reply.address = 1;
        if (test == 33) {
            reply.parameter_bytes = 1;
            reply.arguments = {{0, 1}};
        }
        if (test == 34) {
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
        }
        return reply;
    }

    ExecutionWireReply handle(const ExecutionReceivedRequest& request) {
        auto operation = request.message.header.operation;
        ExecutionPayloadReader input(request.message.payload);
        if (operation == ExecutionOpcode::kHello) {
            auto ownership = ExecutionBootstrapOwnership::kRetained;
            if (decode_execution_context_bootstrap(request.message.payload))
                execution = request.connection;
            else {
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
        if (operation == ExecutionOpcode::kDnnBackendCreate) {
            ExecutionDnnBackendCreate value;
            expect(request.connection == metadata &&
                       read_execution_dnn_backend_create(input, value) && input.empty() &&
                       value.has_output,
                   "descriptors create on unchanged metadata owner");
            auto reply = complete();
            reply.id = 100 + value.kind;
            expect(kinds.emplace(reply.id, value.kind).second, "unique native descriptor identity");
            ++creates;
            return reply;
        }
        if (operation == ExecutionOpcode::kDnnBackendContext) {
            ExecutionDnnBackendContext envelope;
            expect(request.connection == execution &&
                       read_execution_dnn_backend_context(input, envelope) && input.empty() &&
                       envelope.metadata == metadata,
                   "planning frame preserves metadata owner and authenticated execution context");
            ExecutionPayloadReader nested(envelope.payload);
            if (envelope.operation == ExecutionOpcode::kDnnBackendFinalize) {
                std::uint64_t id{};
                expect(
                    !envelope.handle && nested.read_u64(id) && nested.empty() && kinds.at(id) != 16,
                    "planning Finalize has current-context authority without invented handle");
                return complete();
            }
            return attribute(envelope.operation, nested, envelope.handle);
        }
        if (operation == ExecutionOpcode::kDnnBackendPrepare ||
            operation == ExecutionOpcode::kDnnBackendSet ||
            operation == ExecutionOpcode::kDnnBackendGet) {
            expect(request.connection == metadata, "context-independent metadata channel");
            return attribute(operation, input, std::nullopt);
        }
        if (operation == ExecutionOpcode::kPrepareDnnBackendExecute) {
            expect(request.connection == execution, "Execute preparation is not a metadata call");
            return prepare_execute(input);
        }
        if (operation == ExecutionOpcode::kDnnBackendExecute) {
            expect(request.connection == execution, "Execute uses owned execution context");
            return execute(input);
        }
        if (operation == ExecutionOpcode::kCreateDnn) {
            expect(request.connection == execution && input.empty(), "owned handle creation");
            auto reply = complete();
            reply.id = 21 + handle_creates++;
            return reply;
        }
        if (operation == ExecutionOpcode::kAllocate) {
            std::uint64_t bytes{};
            expect(input.read_u64(bytes) && bytes == 64 && input.empty(),
                   "bounded owned allocation");
            auto reply = complete(std::nullopt);
            reply.id = 41;
            reply.address = kAddress;
            return reply;
        }
        if (operation == ExecutionOpcode::kClose) {
            expect(input.empty() && request.connection == execution, "context-first teardown");
            return complete(std::nullopt);
        }
        std::uint64_t id{};
        expect(input.read_u64(id) && input.empty(), "single identity lifecycle/probe");
        if (operation == ExecutionOpcode::kGetDnnStream) {
            expect(id == 21 && request.connection == execution, "healthy execution-only probe");
            ++stream_probes;
        } else if (operation == ExecutionOpcode::kDnnBackendInitialize) {
            expect((id == 105 || id == 115) && request.connection == metadata,
                   "healthy metadata-only reset probe");
            ++metadata_probes;
        } else if (operation == ExecutionOpcode::kDnnBackendFinalize) {
            expect(id == 116 && request.connection == metadata,
                   "VariantPack finalization remains host metadata");
            ++direct_finalizes;
        } else if (operation == ExecutionOpcode::kDnnBackendDestroy) {
            expect(request.connection == metadata && kinds.erase(id) == 1,
                   "context-free descriptor cleanup");
        } else if (operation == ExecutionOpcode::kDestroyDnn) {
            expect(id == 21 || id == 22, "destroy original owned handle");
        } else {
            expect(operation == ExecutionOpcode::kRelease && id == 41,
                   "release exact owned allocation");
            return complete(std::nullopt);
        }
        return complete();
    }
};

void run(const char* provider, unsigned test) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticated process identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-backend-front-" + std::to_string(::getpid()) + ".sock";
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(!::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) &&
               !::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) &&
               !::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1),
           "set local fake endpoint");
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start actual authenticated local transport");
    const auto wanted = expected(test);
    auto client = std::async(std::launch::async, [=] {
        Application app(provider);
        if (test != 35) app.pages.seed(0);
        expect(app.execute() == wanted.status,
               "exact first Execute/preparation containment status");
        if (wanted.metadata && wanted.execution) {
            app.pages.seed(1);
            check(app.execute());
        }
        expect(app.api.initialize(app.descriptors[0]) ==
                   (wanted.metadata ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_INITIALIZED),
               "metadata fault bit does not silently poison or revive execution ownership");
        cudaStream_t stream{};
        expect(cudnnGetStream(app.handles[0], &stream) ==
                   (wanted.execution ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_INITIALIZED),
               "execution fault bit remains independent from descriptor ownership");
        if (wanted.metadata && wanted.execution) app.close();
        expect(!glimmer_dnn_backend_graph_native_calls() && !glimmer_dnn_native_calls(),
               "no direct or fallback original SDK call in application process");
    });
    FakeService fake;
    fake.test = test;
    try {
        glimmer::tests::ExecutionDnnProfileFixture profiles;
        const auto deadline = Clock::now() + std::chrono::seconds{10};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(Clock::now() < deadline && server.poll_for(1), "bounded fake service progress");
            if (auto request = server.take_request()) {
                expect(request->peer == *identity, "exact peer authenticated on every request");
                if (profiles.respond(server, *request)) continue;
                const auto result = fake.handle(*request);
                const auto encoded = encode_execution_reply(result);
                expect(encoded && decode_execution_reply(*encoded),
                       "hostile replies are structurally valid and reach frontend semantic checks");
                auto header = request->message.header;
                header.kind = ExecutionMessageKind::kReply;
                header.session = request->connection;
                header.incarnation = {71, 97};
                expect(server.reply(request->connection, {header, *encoded}),
                       "send exact fake response");
            }
        }
        client.get();
        profiles.verify();
        const auto repeat = wanted.metadata && wanted.execution;
        expect(fake.creates == 5 && fake.handle_creates == 2 && fake.handle_set_errors == 1 &&
                   fake.context_prepares == 6 && fake.context_finals == 6 &&
                   fake.direct_finalizes == 1 && fake.prepares == (repeat ? 2U : 1U) &&
                   fake.finals == (wanted.has_final ? 1U : 0U) + (repeat ? 1U : 0U) &&
                   fake.stream_probes == (wanted.execution ? 1U : 0U) &&
                   fake.metadata_probes == (wanted.metadata ? (repeat ? 2U : 1U) : 0U),
               "exact request counts prove no poisoned retry, hidden SDK query or speculative "
               "Execute");
        if (repeat)
            expect(fake.kinds.empty(),
                   "healthy source identities cleaned after context-first close");
    } catch (...) {
        server.stop();  // Release a blocked client before future destruction on failure.
        if (client.valid()) {
            expect(client.wait_for(std::chrono::seconds{5}) == std::future_status::ready,
                   "client exits promptly after synthetic endpoint shutdown");
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
    expect(!input.empty() && input.size() <= 2, "bounded child case");
    for (const auto digit : input) {
        expect(digit >= '0' && digit <= '9', "decimal child case");
        value = value * 10 + static_cast<unsigned>(digit - '0');
    }
    expect(value < kCases, "known test case");
    return value;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 4, "provider path and optional isolated case");
        if (argc == 4) {
            expect(std::string_view{argv[2]} == "--case", "isolated test option");
            run(argv[1], number(argv[3]));
        } else {
            for (unsigned index = 0; index != kCases; ++index) {
                const auto text = std::to_string(index);
                const auto child = ::fork();
                expect(child >= 0, "fork clean interceptor state");
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
                    throw std::runtime_error("backend frontend case failed: " + text);
            }
        }
        std::cout << "Backend frontend synthetic transport tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
