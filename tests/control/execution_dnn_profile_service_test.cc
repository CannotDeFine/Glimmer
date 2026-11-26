#include "runtime.h"
#include "internal/cuda_dnn_registry.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "internal/process_identity.h"
#include "../backend/cuda_execution_domain_fixture.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
constexpr std::uint32_t kVersion = CUDNN_VERSION;
constexpr std::uint32_t kOtherVersion = kVersion == 8907 ? 91002 : 8907;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Observations {
    std::atomic<unsigned int> sdk_calls{}, tensor_creates{}, tensor_destroys{}, version_queries{};
    std::atomic<unsigned int> unexpected_sdk{}, unexpected_driver{}, context_creates{},
        context_destroys{};
    std::atomic<bool> should_fail_create{};
} observed;
thread_local CUcontext current_context{};

cudnnStatus_t unexpected_dnn() {
    ++observed.sdk_calls;
    ++observed.unexpected_sdk;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
CUresult unexpected_driver() {
    ++observed.unexpected_driver;
    return CUDA_ERROR_NOT_SUPPORTED;
}

// All callbacks use the actual selected SDK headers. No SDK9-only fixture,
// original library function, GPU allocation or compute implementation is used.
CudaDnnDispatch metadata_dispatch() {
    CudaDnnDispatch sdk;
    sdk.tensor_create = [](cudnnTensorDescriptor_t* output) {
        ++observed.sdk_calls;
        if (observed.should_fail_create.exchange(false)) return CUDNN_STATUS_BAD_PARAM;
        *output = reinterpret_cast<cudnnTensorDescriptor_t>(new int(0));
        ++observed.tensor_creates;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.tensor_destroy = [](cudnnTensorDescriptor_t tensor) {
        expect(tensor != nullptr, "destroy an actual owned CPU tensor descriptor");
        ++observed.sdk_calls;
        ++observed.tensor_destroys;
        delete reinterpret_cast<int*>(tensor);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.tensor_set_4d = [](cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, int,
                           int, int) { return unexpected_dnn(); };
    sdk.tensor_set_4d_ex = [](cudnnTensorDescriptor_t, cudnnDataType_t, int, int, int, int, int,
                              int, int, int) { return unexpected_dnn(); };
    sdk.tensor_set_nd = [](cudnnTensorDescriptor_t, cudnnDataType_t, int, const int*, const int*) {
        return unexpected_dnn();
    };
    sdk.tensor_set_nd_ex = [](cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int,
                              const int*) { return unexpected_dnn(); };
    sdk.tensor_get_4d = [](cudnnTensorDescriptor_t, cudnnDataType_t*, int*, int*, int*, int*, int*,
                           int*, int*, int*) { return unexpected_dnn(); };
    sdk.tensor_get_nd = [](cudnnTensorDescriptor_t, int, cudnnDataType_t*, int*, int*, int*) {
        return unexpected_dnn();
    };
    sdk.tensor_size = [](cudnnTensorDescriptor_t, std::size_t*) { return unexpected_dnn(); };
    sdk.activation_create = [](CudaDnnActivation*) { return unexpected_dnn(); };
    sdk.activation_destroy = [](CudaDnnActivation) { return unexpected_dnn(); };
    sdk.activation_set = [](CudaDnnActivation, cudnnActivationMode_t, cudnnNanPropagation_t,
                            double) { return unexpected_dnn(); };
    sdk.activation_get = [](CudaDnnActivation, cudnnActivationMode_t*, cudnnNanPropagation_t*,
                            double*) { return unexpected_dnn(); };
    sdk.version = [] {
        ++observed.sdk_calls;
        ++observed.version_queries;
        return std::size_t{kVersion};
    };
    return sdk;
}

CudaSessionDispatch dispatch(std::uint32_t verified_version) {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        *output = current_context = reinterpret_cast<CUcontext>(new int(0));
        ++observed.context_creates;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext value) {
        current_context = value;
        return CUDA_SUCCESS;
    };
    sdk.context_destroy = [](CUcontext value) {
        expect(value && value == current_context, "destroy only the selected fake native context");
        delete reinterpret_cast<int*>(value);
        current_context = nullptr;
        ++observed.context_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* output) {
        *output = current_context;
        return CUDA_SUCCESS;
    };
    sdk.module_load = [](CUmodule*, const void*) { return unexpected_driver(); };
    sdk.module_unload = [](CUmodule) { return unexpected_driver(); };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) { return unexpected_driver(); };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return unexpected_driver();
    };
    sdk.allocate = [](CUdeviceptr*, std::size_t) { return unexpected_driver(); };
    sdk.release = [](CUdeviceptr) { return unexpected_driver(); };
    sdk.upload = [](CUdeviceptr, const void*, std::size_t) { return unexpected_driver(); };
    sdk.download = [](void*, CUdeviceptr, std::size_t) { return unexpected_driver(); };
    sdk.stream_create = [](CUstream*, unsigned int) { return unexpected_driver(); };
    sdk.stream_destroy = [](CUstream) { return unexpected_driver(); };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) { return unexpected_driver(); };
    sdk.execution.event_create = [](CUevent*, unsigned int) { return unexpected_driver(); };
    sdk.execution.event_record = [](CUevent, CUstream) { return unexpected_driver(); };
    sdk.execution.event_query = [](CUevent) { return unexpected_driver(); };
    sdk.execution.event_destroy = [](CUevent) { return unexpected_driver(); };
    sdk.execution.stream_synchronize = [](CUstream) { return unexpected_driver(); };
    sdk.host_register = [](void*, std::size_t, unsigned int) { return unexpected_driver(); };
    sdk.host_unregister = [](void*) { return unexpected_driver(); };
    sdk.host_map = [](CUdeviceptr*, void*, unsigned int) { return unexpected_driver(); };
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(metadata_dispatch());
    sdk.verified_dnn_header_version = verified_version;
    return sdk;
}

std::vector<std::byte> role(std::uint32_t value) {
    ExecutionPayloadWriter output;
    expect(output.write_u32(value) && output.write_u32(0), "encode exact Hello role");
    return output.take();
}
std::vector<std::byte> profile(std::uint32_t value = kVersion) {
    ExecutionPayloadWriter output;
    expect(write_execution_dnn_profile(output, value), "encode exact profile version");
    return output.take();
}
std::vector<std::byte> tensor_create() {
    ExecutionPayloadWriter output;
    expect(output.write_u32(static_cast<std::uint32_t>(ExecutionDnnObjectKind::kTensor)),
           "encode valid tensor constructor request");
    return output.take();
}
std::vector<std::byte> version_query() {
    ExecutionPayloadWriter output;
    expect(write_execution_dnn_library_info(output, {ExecutionDnnLibraryInfoKind::kVersion, 0, 0}),
           "encode valid library observation");
    return output.take();
}
ExecutionWireReply complete(std::optional<ExecutionWireReply> reply) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kComplete && !reply->status,
           "real service completes the requested operation");
    return std::move(*reply);
}
void reject(std::optional<ExecutionWireReply> reply, CUresult expected) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
               reply->status == expected && !reply->original_status && !reply->id &&
               !reply->address && !reply->parameter_bytes && reply->arguments.empty() &&
               reply->data.empty(),
           "binding policy rejects without SDK provenance, output or capability");
}
void bind(ExecutionSocketClient& client) {
    const auto request = profile();
    const auto reply = complete(client.exchange(ExecutionOpcode::kBindDnnProfile, request));
    expect(!reply.original_status && !reply.id && !reply.address && !reply.parameter_bytes &&
               reply.arguments.empty() && reply.data == request,
           "successful binding echoes the exact canonical four bytes without native provenance");
}
std::uint64_t open(ExecutionSocketClient& client, bool is_metadata) {
    const auto reply =
        complete(is_metadata ? client.open(role(kExecutionDnnMetadataRole)) : client.open());
    const auto ownership = decode_execution_bootstrap_ownership(reply.data);
    expect(ownership == (is_metadata ? ExecutionBootstrapOwnership::kMetadataRetained
                                     : ExecutionBootstrapOwnership::kRetained),
           "Hello reports the actual metadata or ordinary context owner");
    expect(client.session() != 0, "Hello installs one authenticated session epoch");
    return client.session();
}
void close(ExecutionSocketClient& client, bool has_original = true) {
    const auto reply = complete(client.exchange(ExecutionOpcode::kClose));
    expect(
        reply.original_status == (has_original ? std::optional<std::int32_t>{0} : std::nullopt) &&
            !reply.id && !reply.address && !reply.parameter_bytes && reply.arguments.empty() &&
            reply.data.empty(),
        "Close preserves the role's exact cleanup provenance without profile authority");
}
void unbound_sweep(ExecutionSocketClient& client) {
    const auto before = observed.sdk_calls.load();
    unsigned int checked{};
    for (auto value = static_cast<std::uint32_t>(ExecutionOpcode::kCreateDnnMetadata);
         value <= static_cast<std::uint32_t>(ExecutionOpcode::kDnnLegacyCompute); ++value) {
        const auto opcode = static_cast<ExecutionOpcode>(value);
        if (!execution_opcode_is_dnn(opcode)) continue;
        reject(client.exchange(opcode), CUDA_ERROR_NOT_SUPPORTED);
        ++checked;
    }
    expect(checked == 69, "all 69 current DNN opcodes are covered without a silent range shrink");
    // Valid payloads distinguish the binding gate from malformed-argument rejection.
    reject(client.exchange(ExecutionOpcode::kCreateDnnMetadata, tensor_create()),
           CUDA_ERROR_NOT_SUPPORTED);
    reject(client.exchange(ExecutionOpcode::kDnnLibraryInfo, version_query()),
           CUDA_ERROR_NOT_SUPPORTED);
    expect(observed.sdk_calls.load() == before, "unbound sweep cannot reach any fake SDK callback");
}
void malformed_binds(ExecutionSocketClient& client) {
    const auto before = observed.sdk_calls.load();
    const auto canonical = profile();
    for (std::size_t bytes = 0; bytes != canonical.size(); ++bytes)
        reject(client.exchange(ExecutionOpcode::kBindDnnProfile, std::span{canonical}.first(bytes)),
               CUDA_ERROR_INVALID_VALUE);
    const std::array<std::byte, 4> zero{};
    reject(client.exchange(ExecutionOpcode::kBindDnnProfile, zero), CUDA_ERROR_INVALID_VALUE);
    auto trailing = canonical;
    trailing.push_back(std::byte{1});
    reject(client.exchange(ExecutionOpcode::kBindDnnProfile, trailing), CUDA_ERROR_INVALID_VALUE);
    expect(observed.sdk_calls.load() == before, "malformed binding is a CPU-only rejection");
}
void metadata_round_trip(ExecutionSocketClient& client) {
    const auto before = observed.sdk_calls.load();
    const auto created =
        complete(client.exchange(ExecutionOpcode::kCreateDnnMetadata, tensor_create()));
    expect(created.original_status == 0 && created.id && !created.address &&
               !created.parameter_bytes && created.data.empty() && created.arguments.empty(),
           "bound metadata publishes exactly one owned tensor capability");
    const auto version =
        complete(client.exchange(ExecutionOpcode::kDnnLibraryInfo, version_query()));
    expect(!version.original_status && version.address == kVersion && !version.id &&
               !version.parameter_bytes && version.data.empty() && version.arguments.empty(),
           "size-returning version observation preserves actual output without invented status");
    ExecutionPayloadWriter destroy;
    expect(destroy.write_u64(created.id), "encode exact tensor capability for destruction");
    const auto destroyed =
        complete(client.exchange(ExecutionOpcode::kDestroyDnnMetadata, destroy.take()));
    expect(destroyed.original_status == 0 && !destroyed.id && !destroyed.address &&
               !destroyed.parameter_bytes && destroyed.data.empty() && destroyed.arguments.empty(),
           "bound metadata destroys its tensor once with native status intact");
    observed.should_fail_create = true;
    const auto failed = client.exchange(ExecutionOpcode::kCreateDnnMetadata, tensor_create());
    expect(failed && failed->disposition == ExecutionWireDisposition::kComplete &&
               failed->status == CUDNN_STATUS_BAD_PARAM &&
               failed->original_status == static_cast<int>(CUDNN_STATUS_BAD_PARAM) && !failed->id &&
               !failed->address && !failed->parameter_bytes && failed->data.empty() &&
               failed->arguments.empty(),
           "native constructor failure keeps its selected-header status without a capability");
    expect(observed.sdk_calls.load() == before + 4,
           "only create, version, destroy and the declared native failure entered the SDK");
}

template <typename Function>
void drive(ExecutionService& service, Function function) {
    auto client = std::async(std::launch::async, std::move(function));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (client.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded real socket service progress");
    client.get();
}
ExecutionServiceConfig configuration(const std::string& path) {
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.uses_original_kernels = true;
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
    return config;
}
void binding_epochs(const std::string& path, TaskPeerIdentity identity) {
    ExecutionService service(configuration(path), dispatch(kVersion));
    expect(service.start(), "start matching-profile service");
    drive(service, [&] {
        for (const bool is_metadata : {true, false}) {
            std::uint64_t previous{};
            {
                ExecutionSocketClient client({path, identity, 3000});
                previous = open(client, is_metadata);
                unbound_sweep(client);
                malformed_binds(client);
                unbound_sweep(client);
                const auto before = observed.sdk_calls.load();
                bind(client);
                bind(client);
                expect(observed.sdk_calls.load() == before,
                       "matching bind and rebind are SDK-free");
                if (is_metadata) metadata_round_trip(client);
                reject(client.exchange(ExecutionOpcode::kBindDnnProfile, profile(kOtherVersion)),
                       CUDA_ERROR_NOT_SUPPORTED);
                reject(client.exchange(ExecutionOpcode::kBindDnnProfile, profile()),
                       CUDA_ERROR_NOT_SUPPORTED);
                unbound_sweep(client);
                close(client);
            }
            {
                ExecutionSocketClient client({path, identity, 3000});
                expect(open(client, is_metadata) != previous, "new connection has a fresh epoch");
                unbound_sweep(client);
                bind(client);
                close(client);
            }
            {
                ExecutionSocketClient client({path, identity, 3000});
                (void)open(client, is_metadata);
                reject(client.exchange(ExecutionOpcode::kBindDnnProfile, profile(kOtherVersion)),
                       CUDA_ERROR_NOT_SUPPORTED);
                reject(client.exchange(ExecutionOpcode::kBindDnnProfile, profile()),
                       CUDA_ERROR_NOT_SUPPORTED);
                unbound_sweep(client);
                close(client);
            }
        }
    });
    expect(service.request_stop(), "all matching and mismatched epochs close normally");
}
void composition_rejection(const std::string& path, TaskPeerIdentity identity,
                           std::uint32_t verified_version) {
    const auto before = observed.sdk_calls.load();
    ExecutionService service(configuration(path), dispatch(verified_version));
    expect(service.start(), "start service with deliberately unproved composition profile");
    drive(service, [&] {
        for (const bool is_metadata : {true, false}) {
            ExecutionSocketClient client({path, identity, 3000});
            (void)open(client, is_metadata);
            reject(client.exchange(ExecutionOpcode::kBindDnnProfile, profile()),
                   CUDA_ERROR_NOT_SUPPORTED);
            unbound_sweep(client);
            close(client);
        }
    });
    expect(service.request_stop() && observed.sdk_calls.load() == before,
           "unproved service composition cannot enter the DNN SDK");
}
void wrong_roles(const std::string& path, TaskPeerIdentity identity) {
    const auto before = observed.sdk_calls.load();
    ExecutionService service(configuration(path), dispatch(kVersion));
    expect(service.start(), "start role-isolation service");
    drive(service, [&] {
        for (const auto selected : {kExecutionHostOwnerRole, kExecutionPointerMetadataRole}) {
            ExecutionSocketClient client({path, identity, 3000});
            (void)complete(client.open(role(selected)));
            reject(client.exchange(ExecutionOpcode::kBindDnnProfile, profile()),
                   CUDA_ERROR_INVALID_VALUE);
            unbound_sweep(client);
            close(client, false);
        }
    });
    expect(service.request_stop() && observed.sdk_calls.load() == before,
           "non-DNN helper roles cannot acquire DNN profile authority");
}

namespace shared_profile {
namespace domain_fixture = glimmer::test::cuda_domain;
class InertPeer final {
   public:
    InertPeer() {
        int descriptors[2]{};
        expect(::pipe2(descriptors, O_CLOEXEC) == 0, "create owned inert-peer lifetime pipe");
        child_ = ::fork();
        if (child_ < 0) {
            (void)::close(descriptors[0]);
            (void)::close(descriptors[1]);
            throw std::runtime_error("fork owned inert CPU peer");
        }
        if (!child_) {
            if (::close(descriptors[1]) != 0) ::_exit(EXIT_FAILURE);
            char byte{};
            ssize_t received{};
            do {
                received = ::read(descriptors[0], &byte, 1);
            } while (received < 0 && errno == EINTR);
            const bool is_closed = ::close(descriptors[0]) == 0;
            ::_exit(received == 0 && is_closed ? EXIT_SUCCESS : EXIT_FAILURE);
        }
        descriptor_ = descriptors[1];
        if (::close(descriptors[0]) != 0) std::terminate();
    }
    ~InertPeer() {
        const bool is_closed = ::close(descriptor_) == 0;
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child_, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (!is_closed || waited != child_ || !WIFEXITED(status) || WEXITSTATUS(status))
            std::terminate();
    }
    InertPeer(const InertPeer&) = delete;
    InertPeer& operator=(const InertPeer&) = delete;
    TaskPeerIdentity identity() const {
        TaskPeerIdentity result{static_cast<std::uint32_t>(child_),
                                static_cast<std::uint32_t>(::geteuid()), 0};
        expect(read_process_start_time(result.pid, &result.start_time_ticks),
               "read exact owned inert-peer incarnation");
        return result;
    }

   private:
    pid_t child_{};
    int descriptor_{-1};
};
std::uint64_t context_gets{}, context_sets{};
std::vector<std::pair<CUstream, int>> stream_priorities;
CUresult CUDAAPI get_current(CUcontext* output) {
    ++context_gets;
    return domain_fixture::context_get_current(output);
}
CUresult CUDAAPI set_current(CUcontext context) {
    ++context_sets;
    return domain_fixture::context_set_current(context);
}
using Counts = std::array<std::uint64_t, 4>;
Counts counts() {
    return {context_gets, context_sets, domain_fixture::driver.context_control_calls,
            observed.sdk_calls.load()};
}
Counts difference(const Counts& before) {
    auto result = counts();
    for (std::size_t index = 0; index < result.size(); ++index) result[index] -= before[index];
    return result;
}
Counts idle_turn(ExecutionService& service) {
    const auto before = counts();
    expect(service.step(0), "shared quiescent owner turn stays healthy");
    return difference(before);
}
struct Turn {
    ExecutionOpcode operation{};
    unsigned int index{};
    Counts background{}, actual{};
    bool is_ready{};
};
ExecutionWireReply receive(ExecutionService& service, ExecutionSocketClient& client,
                           ExecutionSocketSubmission submitted, ExecutionOpcode operation,
                           std::vector<Turn>* turns = nullptr, Counts background = {}) {
    expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
           "serial client enqueues exactly one authenticated control request");
    for (unsigned int index = 0; index != 256; ++index) {
        const auto before = counts();
        expect(service.step(0) && client.poll_for(0), "bounded serial control transport progress");
        auto reply = client.take_reply(submitted.sequence);
        if (turns)
            turns->push_back({operation, index, background, difference(before), bool(reply)});
        if (reply) return std::move(*reply);
    }
    throw std::runtime_error("shared control reply not ready within 256 structural owner turns");
}
void run(const std::string& path, TaskPeerIdentity identity) {
    InertPeer dormant;
    domain_fixture::driver = {};
    context_gets = context_sets = 0;
    stream_priorities.clear();
    auto domain_callbacks = domain_fixture::configuration_domain_dispatch();
    domain_callbacks.context_get_current = get_current;
    domain_callbacks.context_set_current = set_current;
    auto domain = std::make_shared<CudaExecutionDomainOwner>(
        domain_callbacks, CudaExecutionDomainConfig{99, 0, CU_CTX_MAP_HOST, 4, 512});
    auto sdk = domain_fixture::session_dispatch();
    sdk.execution.context_get_current = get_current;
    sdk.context_set_current = set_current;
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(metadata_dispatch());
    sdk.verified_dnn_header_version = kVersion;
    // Native Driver callback signatures fix these adjacent parameter orders.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    sdk.context.priority_range = [](int* least, int* greatest) {
        *least = 0;
        *greatest = -3;
        return CUDA_SUCCESS;
    };
    sdk.stream_create_priority = [](CUstream* output, unsigned int flags, int priority) {
        const auto status = domain_fixture::stream_create(output, flags);
        if (status == CUDA_SUCCESS) stream_priorities.emplace_back(*output, priority);
        return status;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    sdk.stream_get_priority = [](CUstream stream, int* output) {
        for (const auto& entry : stream_priorities) {
            if (entry.first != stream) continue;
            *output = entry.second;
            return CUDA_SUCCESS;
        }
        return CUDA_ERROR_INVALID_HANDLE;
    };
    auto config = configuration(path);
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kPrioritySpaceShare;
    config.client_priorities = {{identity, 100}, {dormant.identity(), 0}};
    config.shared = ExecutionServiceSharedConfig{domain, true};
    ExecutionService service(config, sdk);
    expect(service.start(), "start actual shared-domain profile-control service");
    ExecutionSocketClient client({path, identity, 3000});
    const auto bootstrap =
        encode_execution_context_bootstrap({ExecutionContextPurpose::kRuntimePrimary, 0, 1});
    expect(bootstrap.has_value(), "encode a genuine shared-domain ordinary context");
    const auto opened =
        complete(receive(service, client, client.submit_open(*bootstrap), ExecutionOpcode::kHello));
    expect(opened.original_status == 0 && domain->snapshot().attached_members == 1 &&
               domain_fixture::driver.context_creates == 1,
           "ordinary client attached to the single actual fake shared context");

    // A quiescent shared slot is scheduled each owner turn even with no request.
    // Compare the exact per-turn callback inventory, not elapsed time or a whole
    // exchange total that would incorrectly attribute background work to Bind.
    (void)idle_turn(service);
    const auto background = idle_turn(service);
    bool is_background_stable = idle_turn(service) == background;
    std::vector<Turn> turns;
    const auto request = [&](ExecutionOpcode operation, std::span<const std::byte> payload) {
        is_background_stable = (idle_turn(service) == background) && is_background_stable;
        return receive(service, client, client.submit(operation, payload), operation, &turns,
                       background);
    };
    ExecutionDnnCompute compute;
    compute.kind = ExecutionDnnComputeKind::kAddTensor;
    compute.handle = 1;
    compute.metadata = 2;
    compute.tensors = {1, 2};
    compute.buffers = {{{1, 0}, {2, 0}}};
    compute.scalar_bytes = 4;
    ExecutionPayloadWriter compute_payload;
    expect(write_execution_dnn_compute(compute_payload, compute),
           "unbound compute request has a fully canonical typed payload");
    const auto computation = compute_payload.take();
    reject(request(ExecutionOpcode::kDnnCompute, computation), CUDA_ERROR_NOT_SUPPORTED);
    reject(request(ExecutionOpcode::kCreateDnnMetadata, tensor_create()), CUDA_ERROR_NOT_SUPPORTED);
    reject(request(ExecutionOpcode::kDnnLibraryInfo, version_query()), CUDA_ERROR_NOT_SUPPORTED);
    const std::array<std::byte, 4> zero{};
    reject(request(ExecutionOpcode::kBindDnnProfile, zero), CUDA_ERROR_INVALID_VALUE);
    const auto canonical = profile();
    const auto bound = complete(request(ExecutionOpcode::kBindDnnProfile, canonical));
    expect(bound.data == canonical && !bound.original_status && !bound.id && !bound.address &&
               !bound.parameter_bytes && bound.arguments.empty(),
           "shared binding echoes the exact profile without SDK output or authority");
    const auto repeated = complete(request(ExecutionOpcode::kBindDnnProfile, canonical));
    expect(repeated.data == canonical && !repeated.original_status,
           "same-profile shared rebind remains an SDK-free idempotent control");
    auto trailing = canonical;
    trailing.push_back(std::byte{});
    reject(request(ExecutionOpcode::kBindDnnProfile, trailing), CUDA_ERROR_INVALID_VALUE);
    reject(request(ExecutionOpcode::kBindDnnProfile, profile(kOtherVersion)),
           CUDA_ERROR_NOT_SUPPORTED);
    reject(request(ExecutionOpcode::kBindDnnProfile, canonical), CUDA_ERROR_NOT_SUPPORTED);
    reject(request(ExecutionOpcode::kDnnCompute, computation), CUDA_ERROR_NOT_SUPPORTED);
    reject(request(ExecutionOpcode::kCreateDnnMetadata, tensor_create()), CUDA_ERROR_NOT_SUPPORTED);

    const auto closed = complete(
        receive(service, client, client.submit(ExecutionOpcode::kClose), ExecutionOpcode::kClose));
    expect(!closed.original_status && !closed.id && !closed.address && !closed.parameter_bytes &&
               closed.data.empty() && closed.arguments.empty(),
           "shared logical Close retires its member without invented physical-destroy status");
    expect(service.request_stop() && domain->snapshot().is_shutdown &&
               !domain->snapshot().is_quarantined && domain_fixture::driver.context_creates == 1 &&
               domain_fixture::driver.context_destroys == 1 &&
               domain_fixture::driver.stream_destroys == 1 &&
               domain_fixture::driver.live_resources() == 0 && !domain_fixture::driver.launches,
           "profile-control regression ends with complete normal shared-domain cleanup");
    // Defer this assertion until after successful cleanup, including against the
    // old implementation whose request-triggered activations produce excess.
    bool has_excess{};
    for (const auto& turn : turns) {
        if (turn.actual == turn.background) continue;
        has_excess = true;
        std::cerr << "profile-control opcode=" << static_cast<std::uint32_t>(turn.operation)
                  << " turn=" << turn.index << " ready=" << turn.is_ready
                  << " callbacks=[get,set,configuration,dnn] background=";
        for (const auto count : turn.background) std::cerr << count << ',';
        std::cerr << " actual=";
        for (const auto count : turn.actual) std::cerr << count << ',';
        std::cerr << '\n';
    }
    expect(is_background_stable && background[0] && !background[3] && !has_excess && !turns.empty(),
           "shared profile control adds zero request-attributed Driver/SDK callbacks to stable "
           "idle turns");
}
}  // namespace shared_profile
}  // namespace

int main() {
    char pattern[] = "/tmp/glimmer-dnn-profile-service-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        const auto identity = current_process_identity();
        expect(identity.has_value(), "obtain exact authenticated service process identity");
        binding_epochs(std::string(directory) + "/epochs.sock", *identity);
        composition_rejection(std::string(directory) + "/zero.sock", *identity, 0);
        composition_rejection(std::string(directory) + "/wrong.sock", *identity, kOtherVersion);
        wrong_roles(std::string(directory) + "/roles.sock", *identity);
        shared_profile::run(std::string(directory) + "/shared.sock", *identity);
        expect(observed.sdk_calls == 4 && observed.tensor_creates == 1 &&
                   observed.tensor_destroys == 1 && observed.version_queries == 1 &&
                   !observed.unexpected_sdk && !observed.unexpected_driver &&
                   observed.context_creates == observed.context_destroys,
               "exact SDK inventory and normal once-only fake native cleanup");
        std::cout << "cuDNN profile real-service binding checks passed for header " << kVersion
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) {
        std::cerr << "profile service fixture private directory was not empty\n";
        result = EXIT_FAILURE;
    }
    return result;
}
