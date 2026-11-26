#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_backend_protocol.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_backend_graph_fixture.h"

#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_backend_graph;
unsigned int unexpected_lt_calls{};

template <typename... Arguments>
cublasStatus_t unexpected_lt(Arguments...) {
    ++unexpected_lt_calls;
    return CUBLAS_STATUS_INTERNAL_ERROR;
}
glimmer::backend::cuda::CudaSessionDispatch dispatch() {
    auto result = fixture::session_dispatch();
    // A fully constructible wrong-role registry distinguishes role rejection
    // from a missing-capability bootstrap failure. No Lt callback may execute.
    result.lt.layout_create = &unexpected_lt;
    result.lt.layout_destroy = &unexpected_lt;
    result.lt.layout_set = &unexpected_lt;
    result.lt.layout_get = &unexpected_lt;
    result.lt.operation_create = &unexpected_lt;
    result.lt.operation_destroy = &unexpected_lt;
    result.lt.operation_set = &unexpected_lt;
    result.lt.operation_get = &unexpected_lt;
    result.lt.preference_create = &unexpected_lt;
    result.lt.preference_destroy = &unexpected_lt;
    result.lt.preference_set = &unexpected_lt;
    result.lt.preference_get = &unexpected_lt;
    return result;
}

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
ExecutionWireReply complete(std::optional<ExecutionWireReply> reply, bool has_original = true) {
    if (!reply || reply->disposition != ExecutionWireDisposition::kComplete || reply->status ||
        reply->parameter_bytes || reply->address || !reply->arguments.empty() ||
        (has_original ? reply->original_status != 0 : reply->original_status.has_value()))
        throw std::runtime_error(
            "metadata success shape: status=" + std::to_string(reply ? reply->status : -1) +
            " original=" + std::to_string(reply ? reply->original_status.value_or(-1) : -1) +
            " flags=" + std::to_string(reply ? reply->parameter_bytes : 0));
    return std::move(*reply);
}
void rejected(std::optional<ExecutionWireReply> reply) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kRejected && reply->status &&
               !reply->original_status && !reply->id && !reply->address &&
               !reply->parameter_bytes && reply->data.empty() && reply->arguments.empty(),
           "rejection has no native provenance, ownership fault or caller output");
}
std::vector<std::byte> identity(std::uint64_t id) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u64(id), "encode identity");
    return writer.take();
}
void open(ExecutionSocketClient& client, std::uint32_t role) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(role) && writer.write_u32(0), "encode metadata role");
    const auto reply = complete(client.open(writer.take()), false);
    expect(decode_execution_bootstrap_ownership(reply.data) ==
               ExecutionBootstrapOwnership::kMetadataRetained,
           "metadata role has no context ownership");
    if (role == kExecutionDnnMetadataRole) {
        ExecutionPayloadWriter binding;
        expect(write_execution_dnn_profile(binding, CUDNN_VERSION), "encode selected DNN profile");
        const auto bound =
            complete(client.exchange(ExecutionOpcode::kBindDnnProfile, binding.take()), false);
        ExecutionPayloadReader reader(bound.data);
        std::uint32_t version{};
        expect(!bound.id && read_execution_dnn_profile(reader, version) &&
                   version == CUDNN_VERSION && reader.empty(),
               "DNN role binds exact headers without native provenance or descriptor authority");
    }
}
std::vector<std::byte> create_request(std::uint32_t kind, std::uint32_t has_output = 1) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_create(writer, {kind, has_output}), "encode creation");
    return writer.take();
}
std::uint64_t create(ExecutionSocketClient& client, std::uint32_t kind) {
    auto reply =
        complete(client.exchange(ExecutionOpcode::kDnnBackendCreate, create_request(kind)));
    expect(reply.id && reply.data.empty(), "creation publishes only an owned identity");
    return reply.id;
}
std::vector<std::byte> prepare_request(ExecutionDnnBackendAttributeOperation operation,
                                       const ExecutionDnnBackendAttributeHeader& header) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_prepare(writer, {operation, header}), "encode preparation");
    return writer.take();
}
ExecutionDnnBackendAttributePreparation prepare(ExecutionSocketClient& client,
                                                ExecutionDnnBackendAttributeOperation operation,
                                                const ExecutionDnnBackendAttributeHeader& header) {
    auto reply = complete(
        client.exchange(ExecutionOpcode::kDnnBackendPrepare, prepare_request(operation, header)),
        false);
    ExecutionDnnBackendAttributePreparation result;
    ExecutionPayloadReader reader(reply.data);
    expect(!reply.id && read_execution_dnn_backend_preparation(reader, result) && reader.empty(),
           "preparation publishes no native receipt or object");
    return result;
}
std::vector<std::byte> attribute(const ExecutionDnnBackendAttribute& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_attribute(writer, value), "encode prepared attribute");
    return writer.take();
}
struct IntegerAttribute {
    std::uint32_t name{};
    std::uint64_t bits{};
};
void set_integer(ExecutionSocketClient& client, std::uint64_t descriptor, IntegerAttribute value) {
    const ExecutionDnnBackendAttributeHeader header{descriptor, value.name, 3, 1, 1};
    const auto prepared = prepare(client, ExecutionDnnBackendAttributeOperation::kSet, header);
    ExecutionPayloadWriter scalar;
    expect(scalar.write_u64(value.bits), "encode exact INT64 scalar bits");
    const auto reply =
        complete(client.exchange(ExecutionOpcode::kDnnBackendSet,
                                 attribute({header, prepared.generation, {3, scalar.take(), {}}})));
    expect(!reply.id && reply.data.empty(), "setter has no output payload");
}
void run_client(const std::string& path, TaskPeerIdentity peer) {
    ExecutionSocketClient first({path, peer, 3000}), second({path, peer, 3000}),
        wrong_role({path, peer, 3000});
    open(first, kExecutionDnnMetadataRole);
    open(second, kExecutionDnnMetadataRole);
    open(wrong_role, kExecutionLtMetadataRole);
    const auto before = fixture::control.callbacks.load();
    for (const auto opcode :
         {ExecutionOpcode::kDnnBackendCreate, ExecutionOpcode::kDnnBackendDestroy,
          ExecutionOpcode::kDnnBackendInitialize, ExecutionOpcode::kDnnBackendFinalize,
          ExecutionOpcode::kDnnBackendPrepare, ExecutionOpcode::kDnnBackendSet,
          ExecutionOpcode::kDnnBackendGet})
        rejected(wrong_role.exchange(opcode, create_request(17)));
    rejected(first.exchange(ExecutionOpcode::kDnnBackendCreate, create_request(17, 0)));
    auto extra = create_request(17);
    extra.push_back(std::byte{});
    rejected(first.exchange(ExecutionOpcode::kDnnBackendCreate, extra));
    expect(fixture::control.callbacks.load() == before, "bad envelopes never call native SDK");

    const auto tensor = create(first, 17), parent = create(first, 22);
    const auto foreign = create(second, 17);
    // Same process and coincident numeric IDs do not confer cross-DNM1 authority.
    expect(tensor == foreign && first.session() != second.session(), "independent owner IDs");
    set_integer(first, tensor, {906, 7});
    complete(first.exchange(ExecutionOpcode::kDnnBackendFinalize, identity(tensor)));
    const ExecutionDnnBackendAttributeHeader child_header{parent, 1610, 15, 1, 1};
    const auto child = prepare(first, ExecutionDnnBackendAttributeOperation::kSet, child_header);
    auto child_request = ExecutionDnnBackendAttribute{
        child_header,
        child.generation,
        {15, {}, {{ExecutionDnnBackendReferenceKind::kDescriptor, second.session(), foreign}}}};
    const auto calls = fixture::control.callbacks.load();
    rejected(first.exchange(ExecutionOpcode::kDnnBackendSet, attribute(child_request)));
    expect(fixture::control.callbacks.load() == calls, "foreign owner rejected before native call");
    child_request.values.references[0].owner = first.session();
    complete(first.exchange(ExecutionOpcode::kDnnBackendSet, attribute(child_request)));
    complete(first.exchange(ExecutionOpcode::kDnnBackendInitialize, identity(tensor)));
    set_integer(first, tensor, {906, 107});
    complete(first.exchange(ExecutionOpcode::kDnnBackendDestroy, identity(tensor)));
    complete(first.exchange(ExecutionOpcode::kDnnBackendFinalize, identity(parent)));
    const auto target = create(first, 17);
    const ExecutionDnnBackendAttributeHeader output_header{parent, 1610, 15, 1, 3};
    const auto output = prepare(first, ExecutionDnnBackendAttributeOperation::kGet, output_header);
    auto copied = complete(first.exchange(
        ExecutionOpcode::kDnnBackendGet,
        attribute({output_header,
                   output.generation,
                   {15,
                    {},
                    {{ExecutionDnnBackendReferenceKind::kDescriptor, first.session(), target}}}})));
    ExecutionDnnBackendGetResult copy_result;
    ExecutionPayloadReader copy_reader(copied.data);
    expect(read_execution_dnn_backend_get_result(copy_reader, copy_result) && copy_reader.empty() &&
               copy_result.count_bytes == 8 && copy_result.count_bits == 1 &&
               copy_result.values.references.empty() && copy_result.values.scalars.empty(),
           "descriptor-output Get never returns or replaces native handles");
    complete(first.exchange(ExecutionOpcode::kDnnBackendDestroy, identity(parent)));
    const ExecutionDnnBackendAttributeHeader uid_header{target, 906, 3, 1, 3};
    const auto uid = prepare(first, ExecutionDnnBackendAttributeOperation::kGet, uid_header);
    expect(uid.output_elements == 1 && uid.output_bytes == 8 && uid.count_bytes == 8,
           "exact scalar prefix preparation");
    auto value = complete(first.exchange(ExecutionOpcode::kDnnBackendGet,
                                         attribute({uid_header, uid.generation, {3, {}, {}}})));
    ExecutionDnnBackendGetResult returned;
    ExecutionPayloadReader value_reader(value.data);
    expect(read_execution_dnn_backend_get_result(value_reader, returned) && value_reader.empty(),
           "complete typed Get result");
    ExecutionPayloadReader scalar_reader(returned.values.scalars);
    std::uint64_t bits{};
    expect(scalar_reader.read_u64(bits) && scalar_reader.empty() && bits == 7 &&
               returned.count_bytes == 8 && returned.count_bits == 1,
           "copied child retains original value after source reset and both owners destroyed");
    complete(first.exchange(ExecutionOpcode::kDnnBackendInitialize, identity(target)));
    rejected(first.exchange(ExecutionOpcode::kDnnBackendGet,
                            attribute({uid_header, uid.generation, {3, {}, {}}})));

    fixture::control.set_status.store(CUDNN_STATUS_BAD_PARAM);
    const ExecutionDnnBackendAttributeHeader dirty_header{target, 906, 3, 1, 1};
    const auto dirty = prepare(first, ExecutionDnnBackendAttributeOperation::kSet, dirty_header);
    ExecutionPayloadWriter dirty_scalar;
    expect(dirty_scalar.write_u64(8), "encode failed native setter input");
    auto failure =
        first.exchange(ExecutionOpcode::kDnnBackendSet,
                       attribute({dirty_header, dirty.generation, {3, dirty_scalar.take(), {}}}));
    expect(failure && failure->disposition == ExecutionWireDisposition::kComplete &&
               failure->original_status == CUDNN_STATUS_BAD_PARAM &&
               failure->status == CUDNN_STATUS_BAD_PARAM && !failure->parameter_bytes &&
               failure->data.empty(),
           "issued configuration failure does not quarantine native ownership");
    fixture::control.set_status.store(CUDNN_STATUS_SUCCESS);
    complete(first.exchange(ExecutionOpcode::kDnnBackendInitialize, identity(target)));
    set_integer(first, target, {906, 19});
    complete(first.exchange(ExecutionOpcode::kDnnBackendDestroy, identity(target)));
    complete(second.exchange(ExecutionOpcode::kDnnBackendDestroy, identity(foreign)));
    complete(first.exchange(ExecutionOpcode::kClose));
    complete(second.exchange(ExecutionOpcode::kClose));
    // Empty Lt shutdown is framework cleanup, not an original library call.
    complete(wrong_role.exchange(ExecutionOpcode::kClose), false);
}
void run(const std::string& path, TaskPeerIdentity peer, ExecutionSchedulingMode mode) {
    fixture::reset();
    unexpected_lt_calls = 0;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{peer, 7}};
    ExecutionService service(config, dispatch());
    expect(service.start(), "start isolated raw metadata service");
    auto client = std::async(std::launch::async, [&] { run_client(path, peer); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{8};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded raw metadata exchange");
    client.get();
    expect(service.request_stop() && fixture::alive() == 0 &&
               fixture::state.creates == fixture::state.destroys &&
               fixture::foundation::control.driver_calls.load() == 0 && !unexpected_lt_calls,
           "all native descriptors released without CUDA, profiling or execution");
}
}  // namespace

int main() {
    char pattern[] = "/tmp/glimmer-backend-graph-service-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        const auto peer = current_process_identity();
        expect(peer.has_value(), "authenticated process identity");
        for (const auto mode : {ExecutionSchedulingMode::kNaive, ExecutionSchedulingMode::kProfile,
                                ExecutionSchedulingMode::kPriority})
            run(std::string(directory) + "/metadata.sock", *peer, mode);
        std::cout << "Backend graph raw service ownership checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
