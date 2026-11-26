#include "internal/cuda_native_execution_lanes.h"
#include "internal/cuda_nccl_native_call.h"

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
using Operation = CudaNcclNativeOperation;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Captured {
    Operation operation{};
    bool has_output{}, has_config{};
    int ranks{}, rank{}, root{};
    std::size_t count{};
    ncclDataType_t datatype{};
    ncclRedOp_t reduction{};
    ncclComm_t communicator{};
    const void* send{};
    void* receive{};
    cudaStream_t stream{};
    std::array<std::byte, 128> unique_id{};
    std::array<std::byte, 88> config{};
    std::optional<std::string> net_name, comm_name;
    const ncclConfig_t* original_config{};
};
struct Fixture {
    std::array<Captured, 64> calls{};
    std::size_t count{}, pending_count{}, initializations{};
    std::array<ncclComm_t*, 8> pending{};
    std::array<const ncclConfig_t*, 8> pending_configs{};
    std::array<char, 8> native_objects{};
    char context{}, stream{}, input{}, output{};
    ncclResult_t status{ncclSuccess};
    bool is_grouped{}, partial_output{}, throws{}, loses_context{}, no_output{}, equal_output{};
    std::weak_ptr<CudaNcclNativeCall> reentry;
    std::thread::id worker;
    void reset() {
        *this = {};
    }
    Captured& visit(Operation operation) {
        expect(count < calls.size(), "bounded fake native call inventory");
        if (worker == std::thread::id{}) worker = std::this_thread::get_id();
        expect(worker == std::this_thread::get_id(), "all requested callbacks retain one worker");
        auto& result = calls[count++];
        result.operation = operation;
        if (auto packet = reentry.lock()) {
            expect(packet->was_invoked() && !packet->requested_status(),
                   "native callback entry is distinct from its return");
            expect(packet->invoke() == CUDA_ERROR_NOT_READY && !packet->snapshot_after_quiescence(),
                   "native callback reentry cannot issue or read live output cells");
        }
        if (throws) throw std::runtime_error("injected trusted callback exception");
        return result;
    }
    ncclComm_t native(std::size_t index = 0) {
        return reinterpret_cast<ncclComm_t>(&native_objects.at(index));
    }
} fixture;
thread_local CUcontext current_context{};

CUresult CUDAAPI set_current(CUcontext context) {
    current_context = context;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_current(CUcontext* output) {
    if (fixture.loses_context && fixture.count) return CUDA_ERROR_INVALID_CONTEXT;
    *output = current_context;
    return CUDA_SUCCESS;
}
ncclResult_t get_unique_id(ncclUniqueId* output) {
    auto& captured = fixture.visit(Operation::kGetUniqueId);
    captured.has_output = output != nullptr;
    if (output && !fixture.no_output) {
        if (fixture.equal_output) {
            std::memset(output, 0xa5, sizeof(*output));
        } else if (fixture.partial_output) {
            const std::array<std::byte, 3> bytes{std::byte{0xa5}, std::byte{0x11}, std::byte{0xa5}};
            std::memcpy(output, bytes.data(), bytes.size());
        } else {
            std::array<std::byte, 128> bytes{};
            for (std::size_t index = 0; index != bytes.size(); ++index)
                bytes[index] = static_cast<std::byte>((index * 3 + 7) & 255U);
            std::memcpy(output, bytes.data(), bytes.size());
        }
    }
    return fixture.status;
}
void capture_init(Captured& captured, ncclComm_t* output, int ranks, ncclUniqueId id, int rank,
                  const ncclConfig_t* config) {
    captured.has_output = output != nullptr;
    captured.ranks = ranks;
    captured.rank = rank;
    std::memcpy(captured.unique_id.data(), &id, captured.unique_id.size());
    captured.has_config = config != nullptr;
    if (config) {
        std::memcpy(captured.config.data(), config, captured.config.size());
        if (config->netName) captured.net_name = config->netName;
        if (config->commName) captured.comm_name = config->commName;
        captured.original_config = config;
    }
    if (output && !fixture.no_output) {
        if (fixture.equal_output) {
            *output = nullptr;
        } else if (fixture.is_grouped) {
            expect(fixture.pending_count < fixture.pending.size(), "bounded deferred outputs");
            fixture.pending[fixture.pending_count] = output;
            fixture.pending_configs[fixture.pending_count++] = config;
        } else {
            *output = fixture.native(fixture.initializations++);
        }
    }
}
ncclResult_t init_rank(ncclComm_t* output, int ranks, ncclUniqueId id, int rank) {
    auto& captured = fixture.visit(Operation::kInitRank);
    capture_init(captured, output, ranks, id, rank, nullptr);
    return fixture.status;
}
ncclResult_t init_rank_config(ncclComm_t* output, int ranks, ncclUniqueId id, int rank,
                              ncclConfig_t* config) {
    auto& captured = fixture.visit(Operation::kInitRankConfig);
    capture_init(captured, output, ranks, id, rank, config);
    return fixture.status;
}
ncclResult_t get_async_error(ncclComm_t communicator, ncclResult_t* output) {
    auto& captured = fixture.visit(Operation::kGetAsyncError);
    captured.communicator = communicator;
    captured.has_output = output != nullptr;
    if (output && !fixture.no_output) {
        if (fixture.equal_output) {
            std::memset(output, 0xa5, sizeof(*output));
        } else if (fixture.partial_output) {
            const std::array<std::byte, 3> bytes{std::byte{0x11}, std::byte{0xa5}, std::byte{0xa5}};
            std::memcpy(output, bytes.data(), bytes.size());
        } else {
            const std::int32_t raw = -17;
            std::memcpy(output, &raw, sizeof(raw));
        }
    }
    return fixture.status;
}
ncclResult_t destroy(ncclComm_t communicator) {
    fixture.visit(Operation::kDestroy).communicator = communicator;
    return fixture.status;
}
ncclResult_t abort_communicator(ncclComm_t communicator) {
    fixture.visit(Operation::kAbort).communicator = communicator;
    return fixture.status;
}
ncclResult_t group_start() {
    (void)fixture.visit(Operation::kGroupStart);
    fixture.is_grouped = true;
    return fixture.status;
}
ncclResult_t group_end() {
    (void)fixture.visit(Operation::kGroupEnd);
    for (std::size_t index = 0; index != fixture.pending_count; ++index) {
        if (const auto* config = fixture.pending_configs[index]) {
            expect(config->netName && std::string_view{config->netName} == "owned-network" &&
                       config->commName && std::string_view{config->commName} == "owned-name",
                   "group-delayed native reads still see owned config and both strings");
        }
        *fixture.pending[index] = fixture.native(fixture.initializations++);
    }
    fixture.pending_count = 0;
    fixture.is_grouped = false;
    return fixture.status;
}
// NCCL ABI fixes adjacent send/receive pointer parameters.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
ncclResult_t all_gather(const void* send, void* receive, std::size_t count, ncclDataType_t type,
                        ncclComm_t communicator, cudaStream_t stream) {
    auto& captured = fixture.visit(Operation::kAllGather);
    captured.send = send;
    captured.receive = receive;
    captured.count = count;
    captured.datatype = type;
    captured.communicator = communicator;
    captured.stream = stream;
    return fixture.status;
}
// NCCL ABI fixes adjacent send/receive pointer parameters.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
ncclResult_t all_reduce(const void* send, void* receive, std::size_t count, ncclDataType_t type,
                        ncclRedOp_t reduction, ncclComm_t communicator, cudaStream_t stream) {
    auto& captured = fixture.visit(Operation::kAllReduce);
    captured.send = send;
    captured.receive = receive;
    captured.count = count;
    captured.datatype = type;
    captured.reduction = reduction;
    captured.communicator = communicator;
    captured.stream = stream;
    return fixture.status;
}
ncclResult_t bcast(void* buffer, std::size_t count, ncclDataType_t type, int root,
                   ncclComm_t communicator, cudaStream_t stream) {
    auto& captured = fixture.visit(Operation::kBcast);
    captured.receive = buffer;
    captured.count = count;
    captured.datatype = type;
    captured.root = root;
    captured.communicator = communicator;
    captured.stream = stream;
    return fixture.status;
}
CudaNcclNativeDispatch dispatch() {
    return {
        get_unique_id, init_rank, init_rank_config, get_async_error, destroy, abort_communicator,
        group_start,   group_end, all_gather,       all_reduce,      bcast};
}
CudaNcclNativeArguments arguments(Operation operation) {
    CudaNcclNativeArguments result;
    result.operation = operation;
    result.has_output = true;
    for (std::size_t index = 0; index != result.unique_id.size(); ++index)
        result.unique_id[index] = static_cast<std::byte>((index * 5 + 9) & 255U);
    result.ranks = 3;
    result.rank = 2;
    result.root = -9;
    result.count = 127;
    result.datatype = ncclFloat64;
    result.reduction = ncclMax;
    result.communicator = {fixture.native(), std::make_shared<int>(1)};
    result.send = &fixture.input;
    result.receive = &fixture.output;
    result.stream = reinterpret_cast<cudaStream_t>(&fixture.stream);
    result.resources_lifetime = std::make_shared<int>(2);
    return result;
}
CudaNcclNativeConfig full_config() {
    CudaNcclNativeConfig result;
    result.declared_bytes = 88;
    result.magic = 0xcafebeef;
    result.version = 22903;
    for (std::size_t index = 0; index != result.fields.size(); ++index)
        result.fields[index] = -100 - static_cast<std::int32_t>(index);
    result.net_name = "owned-network";
    result.comm_name = "owned-name";
    return result;
}
std::shared_ptr<CudaNcclNativeCall> packet(CudaNcclNativeArguments request) {
    auto created = CudaNcclNativeCall::create(dispatch(), std::move(request));
    expect(created.status == CUDA_SUCCESS && created.call && !created.call->was_invoked() &&
               !created.call->requested_status() && !created.call->snapshot_after_quiescence(),
           "packet owns inputs without pretending native invocation or outputs");
    return std::move(created.call);
}

struct Lane {
    CudaNativeExecutionLanes lanes{{set_current, get_current}};
    std::shared_ptr<const void> lifetime{std::make_shared<int>(3)};
    std::uint64_t sequence{};
    CudaNativeExecutionTicket submit(std::shared_ptr<const void> owner,
                                     std::function<CUresult()> invoke) {
        auto accepted = lanes.submit({1, {7, 11}},
                                     {++sequence,
                                      0,
                                      {1, reinterpret_cast<CUcontext>(&fixture.context), lifetime},
                                      std::move(owner),
                                      std::move(invoke)});
        expect(accepted.status == CUDA_SUCCESS, "actual stable lane admits trusted packet work");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        for (;;) {
            const auto observed = accepted.ticket.observe();
            if (observed.phase == CudaNativeExecutionCallPhase::kReturned ||
                observed.phase == CudaNativeExecutionCallPhase::kUncertain)
                return accepted.ticket;
            expect(std::chrono::steady_clock::now() < deadline, "bounded fake native lane return");
            std::this_thread::yield();
        }
    }
    void invoke(const std::shared_ptr<CudaNcclNativeCall>& call) {
        const auto ticket = submit(call, [call] { return call->invoke(); });
        const auto observed = ticket.observe();
        expect(observed.phase == CudaNativeExecutionCallPhase::kReturned &&
                   observed.status == CUDA_SUCCESS && observed.was_invoked,
               "generic carrier completes actual packet invocation");
        expect(call->was_invoked() && call->requested_status() == fixture.status,
               "owner can read exact requested status after worker return");
        expect(!call->snapshot_after_quiescence(), "owner cannot read worker output cells");
        expect(lanes.release(ticket) == CUDA_SUCCESS,
               "host ticket releases without releasing separately retained packet storage");
    }
    CudaNcclNativeObservation snapshot(const std::shared_ptr<CudaNcclNativeCall>& call) {
        auto output = std::make_shared<std::optional<CudaNcclNativeObservation>>();
        const auto ticket = submit(call, [call, output] {
            *output = call->snapshot_after_quiescence();
            return CUDA_SUCCESS;
        });
        expect(ticket.observe().phase == CudaNativeExecutionCallPhase::kReturned && *output,
               "later callback on the same lane can snapshot proven-quiescent outputs");
        expect(lanes.release(ticket) == CUDA_SUCCESS, "release exact snapshot callback ticket");
        return **output;
    }
    void close() {
        expect(lanes.request_stop() == CUDA_SUCCESS, "stop the empty stable-lane admission");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        for (;;) {
            const auto result = lanes.shutdown();
            if (result == CUDA_SUCCESS) break;
            expect(result == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline,
                   "bounded native lane shutdown joins only proven worker exits");
            std::this_thread::yield();
        }
        expect(lanes.retained_jobs() == 0, "all known host packet tickets released");
    }
};

void all_eleven_on_stable_lane() {
    fixture.reset();
    Lane lane;
    constexpr std::array order{
        Operation::kGetUniqueId,    Operation::kGroupStart, Operation::kInitRank,
        Operation::kInitRankConfig, Operation::kGroupEnd,   Operation::kGetAsyncError,
        Operation::kAllGather,      Operation::kAllReduce,  Operation::kBcast,
        Operation::kDestroy,        Operation::kAbort};
    std::vector<std::shared_ptr<CudaNcclNativeCall>> calls;
    calls.reserve(order.size());
    for (const auto operation : order) {
        auto request = arguments(operation);
        if (operation == Operation::kInitRankConfig) request.config = full_config();
        auto call = packet(request);
        request.unique_id.fill(std::byte{0});
        if (request.config) {
            request.config->net_name = "changed-client-network";
            request.config->comm_name = "changed-client-name";
        }
        lane.invoke(call);
        calls.push_back(std::move(call));
    }
    expect(fixture.count == order.size() && fixture.worker != std::this_thread::get_id(),
           "exact eleven real typed callbacks ran on a distinct stable worker");
    for (std::size_t index = 0; index != order.size(); ++index)
        expect(fixture.calls[index].operation == order[index], "exact typed callback order");
    const auto uid = lane.snapshot(calls[0]);
    for (std::size_t index = 0; index != uid.unique_id.size(); ++index) {
        const auto value = static_cast<std::byte>((index * 3 + 7) & 255U);
        expect(uid.unique_id[index] == value, "complete original UID bytes");
        expect(((std::to_integer<unsigned>(uid.unique_id_changed[index / 8]) >> (index % 8)) &
                1U) == static_cast<unsigned>(value != std::byte{0xa5}),
               "UID difference evidence does not invent equal-value writes");
    }
    for (std::size_t index = 0; index != 2; ++index) {
        const auto initialized = lane.snapshot(calls[index + 2]);
        const auto expected = fixture.native(index);
        const ncclComm_t null_communicator{};
        std::array<std::byte, sizeof(ncclComm_t)> bytes{}, null_bytes{};
        std::memcpy(bytes.data(), &expected, bytes.size());
        std::memcpy(null_bytes.data(), &null_communicator, null_bytes.size());
        std::uint32_t mask{};
        for (std::size_t byte = 0; byte != bytes.size(); ++byte)
            if (bytes[byte] != null_bytes[byte]) mask |= 1U << byte;
        expect(initialized.communicator == expected && initialized.communicator_changed == mask,
               "durable init cells expose exact later outer-End values and byte differences");
    }
    expect(lane.snapshot(calls[5]).async_error == -17 &&
               lane.snapshot(calls[5]).async_error_changed == 15,
           "raw signed async bytes are independent from requested NCCL success");
    const auto expected_id = arguments(Operation::kInitRank).unique_id;
    expect(fixture.calls[2].unique_id == expected_id && fixture.calls[3].unique_id == expected_id &&
               fixture.calls[2].ranks == 3 && fixture.calls[2].rank == 2,
           "UID is copied by value and exact ranks reach both initialization ABIs");
    expect(
        fixture.calls[3].net_name == "owned-network" && fixture.calls[3].comm_name == "owned-name",
        "both config strings are independent of later caller mutations");
    for (std::size_t index = 6; index != 9; ++index) {
        const auto& captured = fixture.calls[index];
        expect(captured.receive == &fixture.output && captured.count == 127 &&
                   captured.datatype == ncclFloat64 && captured.communicator == fixture.native() &&
                   captured.stream == reinterpret_cast<cudaStream_t>(&fixture.stream),
               "all collective ABI pointer/count/type/communicator/stream positions are exact");
        if (index != 8) expect(captured.send == &fixture.input, "exact collective send pointer");
    }
    expect(fixture.calls[7].reduction == ncclMax && fixture.calls[8].root == -9,
           "reduction and signed root are not silently normalized");
    const auto duplicate = lane.submit(calls[0], [call = calls[0]] {
        expect(call->invoke() == CUDA_ERROR_INVALID_VALUE, "packet is not replayable");
        return CUDA_SUCCESS;
    });
    expect(lane.lanes.release(duplicate) == CUDA_SUCCESS && fixture.count == 11,
           "duplicate packet invocation issues no second native call");
    lane.close();
    calls.clear();
    expect(fixture.count == 11, "packet/lane destruction issues no implicit NCCL cleanup");
}

void exact_errors_presence_and_reentry() {
    for (int status = 0; status != 8; ++status) {
        for (const auto operation : {Operation::kGetUniqueId, Operation::kInitRank}) {
            fixture.reset();
            fixture.status = static_cast<ncclResult_t>(status);
            auto call = packet(arguments(operation));
            expect(call->invoke() == CUDA_SUCCESS && call->requested_status() == status,
                   "every original status survives independently of successful carrier");
            const auto output = call->snapshot_after_quiescence();
            expect(output && output->has_output,
                   "native output presence does not depend on requested success");
            if (operation == Operation::kInitRank)
                expect(output->communicator == fixture.native(),
                       "native-error initialization can still publish a nonnull owner");
            call.reset();
            expect(fixture.count == 1,
                   "neither native error nor partial ownership triggers retry or cleanup");
        }
    }
    for (std::uint32_t raw = 1; raw <= 11; ++raw) {
        fixture.reset();
        fixture.status = ncclInProgress;
        auto request = arguments(static_cast<Operation>(raw));
        request.has_output = false;
        request.ranks = -19;
        request.rank = std::numeric_limits<std::int32_t>::min();
        request.count = std::numeric_limits<std::uint64_t>::max();
        request.datatype = ncclNumTypes;
        request.reduction = ncclNumOps;
        request.communicator = {};
        request.send = nullptr;
        request.receive = nullptr;
        request.stream = nullptr;
        auto call = packet(request);
        fixture.reentry = call;
        expect(call->invoke() == CUDA_SUCCESS && call->requested_status() == 7,
               "InProgress is actual requested NCCL status, not a Driver error or terminal proof");
        const auto observed = call->snapshot_after_quiescence();
        expect(observed && !observed->has_output && observed->communicator_changed == 0 &&
                   observed->async_error_changed == 0,
               "absent outputs remain absent independently of native status");
        expect(fixture.count == 1 && !fixture.calls[0].has_output,
               "each nullable ABI is invoked once with actual absent outputs");
        if (raw == 2 || raw == 3)
            expect(fixture.calls[0].ranks == -19 &&
                       fixture.calls[0].rank == std::numeric_limits<std::int32_t>::min(),
                   "signed invalid rank arguments remain native validation inputs");
        if (raw == 3)
            expect(!fixture.calls[0].has_config, "native NULL config remains NULL, not defaults");
        if (raw >= 9)
            expect(fixture.calls[0].count == std::numeric_limits<std::size_t>::max() &&
                       fixture.calls[0].datatype == ncclNumTypes &&
                       fixture.calls[0].communicator == nullptr,
                   "trusted ABI packet does not truncate raw count or normalize SDK sentinel");
        call.reset();
        expect(fixture.count == 1, "no packet destructor native callback");
    }
    fixture.reset();
    fixture.status = ncclRemoteError;
    fixture.partial_output = true;
    auto uid = packet(arguments(Operation::kGetUniqueId));
    expect(uid->invoke() == CUDA_SUCCESS && uid->requested_status() == ncclRemoteError,
           "error status survives partial UID output");
    const auto observed = uid->snapshot_after_quiescence();
    expect(observed && observed->unique_id_changed[0] == std::byte{2} &&
               observed->unique_id[0] == std::byte{0xa5} &&
               observed->unique_id[1] == std::byte{0x11},
           "error byte differences preserve changed bytes but do not claim poison-equal writes");
    auto async = packet(arguments(Operation::kGetAsyncError));
    expect(async->invoke() == CUDA_SUCCESS && async->requested_status() == ncclRemoteError,
           "requested async-query error remains separate from raw output effects");
    const auto changed = async->snapshot_after_quiescence();
    expect(changed && changed->async_error_changed == 1 &&
               std::bit_cast<std::uint32_t>(changed->async_error) == 0xa5a5a511U,
           "partial async enum storage is read as raw bytes, never invalid enum lvalues");
    fixture.reset();
    fixture.throws = true;
    auto throwing = packet(arguments(Operation::kGetUniqueId));
    expect(throwing->invoke() == CUDA_ERROR_UNKNOWN && throwing->was_invoked() &&
               !throwing->requested_status() && !throwing->snapshot_after_quiescence() &&
               throwing->invoke() == CUDA_ERROR_INVALID_VALUE && fixture.count == 1,
           "throwing trusted native callback cannot fabricate returned status or replay");
}

void missing_callbacks_and_authority() {
    fixture.reset();
    for (std::uint32_t raw = 1; raw <= 11; ++raw) {
        auto callbacks = dispatch();
        switch (static_cast<Operation>(raw)) {
            case Operation::kGetUniqueId:
                callbacks.get_unique_id = nullptr;
                break;
            case Operation::kInitRank:
                callbacks.init_rank = nullptr;
                break;
            case Operation::kInitRankConfig:
                callbacks.init_rank_config = nullptr;
                break;
            case Operation::kGetAsyncError:
                callbacks.get_async_error = nullptr;
                break;
            case Operation::kDestroy:
                callbacks.destroy = nullptr;
                break;
            case Operation::kAbort:
                callbacks.abort = nullptr;
                break;
            case Operation::kGroupStart:
                callbacks.group_start = nullptr;
                break;
            case Operation::kGroupEnd:
                callbacks.group_end = nullptr;
                break;
            case Operation::kAllGather:
                callbacks.all_gather = nullptr;
                break;
            case Operation::kAllReduce:
                callbacks.all_reduce = nullptr;
                break;
            case Operation::kBcast:
                callbacks.bcast = nullptr;
                break;
        }
        const auto rejected =
            CudaNcclNativeCall::create(callbacks, arguments(static_cast<Operation>(raw)));
        expect(rejected.status == CUDA_ERROR_NOT_SUPPORTED && !rejected.call,
               "every missing requested callback rejects before SDK or output publication");
    }
    for (const auto raw : {0U, 12U, UINT32_MAX}) {
        const auto rejected =
            CudaNcclNativeCall::create(dispatch(), arguments(static_cast<Operation>(raw)));
        expect(rejected.status == CUDA_ERROR_INVALID_VALUE && !rejected.call,
               "unknown operation cannot create an executable packet");
    }
    auto unowned = arguments(Operation::kDestroy);
    unowned.communicator.lifetime.reset();
    expect(CudaNcclNativeCall::create(dispatch(), unowned).status == CUDA_ERROR_INVALID_HANDLE,
           "raw nonnull communicator is not authority without immutable native-owner lease");
    for (const auto operation : {Operation::kAllGather, Operation::kAllReduce, Operation::kBcast}) {
        auto request = arguments(operation);
        request.resources_lifetime.reset();
        expect(CudaNcclNativeCall::create(dispatch(), request).status == CUDA_ERROR_INVALID_VALUE,
               "all three collectives require prevalidated resource lifetime");
    }
    expect(fixture.count == 0, "all constructor policy rejections have no native provenance");
}

void config_prefixes_and_lifetimes() {
    constexpr std::array<std::size_t, 16> offsets{8,  12, 16, 20, 24, 28, 40, 44,
                                                  56, 60, 64, 68, 72, 76, 80, 84};
    for (std::uint64_t prefix = 0; prefix <= 88; ++prefix) {
        fixture.reset();
        auto request = arguments(Operation::kInitRankConfig);
        auto config = full_config();
        config.declared_bytes = prefix;
        std::array<std::uint32_t, 16> scalars{};
        scalars[0] = config.magic;
        scalars[1] = config.version;
        for (std::size_t index = 0; index != config.fields.size(); ++index)
            scalars[index + 2] = std::bit_cast<std::uint32_t>(config.fields[index]);
        for (std::size_t index = 0; index != scalars.size(); ++index) {
            std::array<std::byte, 4> bytes{};
            std::memcpy(bytes.data(), &scalars[index], bytes.size());
            for (std::size_t byte = 0; byte != bytes.size(); ++byte)
                if (offsets[index] + byte >= prefix) bytes[byte] = std::byte{0};
            std::memcpy(&scalars[index], bytes.data(), bytes.size());
        }
        config.magic = scalars[0];
        config.version = scalars[1];
        for (std::size_t index = 0; index != config.fields.size(); ++index)
            config.fields[index] = std::bit_cast<std::int32_t>(scalars[index + 2]);
        if (prefix < 40) config.net_name.reset();
        if (prefix < 56) config.comm_name.reset();
        request.config = config;
        const auto created = CudaNcclNativeCall::create(dispatch(), request);
        if ((prefix >= 33 && prefix <= 39) || (prefix >= 49 && prefix <= 55)) {
            expect(
                created.status == CUDA_ERROR_INVALID_VALUE && !created.call && fixture.count == 0,
                "every cut native pointer prefix rejects before callback");
            continue;
        }
        expect(created.call && created.call->invoke() == CUDA_SUCCESS,
               "every safe partial scalar/native prefix invokes original ABI");
        const auto& captured = fixture.calls[0];
        std::uint64_t size{};
        std::memcpy(&size, captured.config.data(), sizeof(size));
        expect(size == prefix && captured.has_config && captured.net_name == config.net_name &&
                   captured.comm_name == config.comm_name,
               "declared size and both independent optional strings preserve exact semantics");
        for (std::size_t index = 0; index != offsets.size(); ++index) {
            std::uint32_t actual{};
            std::memcpy(&actual, captured.config.data() + offsets[index], sizeof(actual));
            expect(actual == scalars[index], "exact materialized scalar byte prefix");
        }
    }
    fixture.reset();
    auto request = arguments(Operation::kInitRankConfig);
    request.config = full_config();
    request.config->declared_bytes = UINT64_MAX;
    request.config->net_name = "";
    request.config->comm_name = std::string(kCudaNcclNativeConfigStringLimit, 'x');
    auto owned = std::make_shared<int>(71);
    const std::weak_ptr<int> weak = owned;
    request.communicator.lifetime = owned;
    request.resources_lifetime = owned;
    auto call = packet(std::move(request));
    owned.reset();
    expect(!weak.expired() && call->invoke() == CUDA_SUCCESS,
           "packet retains immutable input owners through native call return");
    expect(fixture.calls[0].net_name == "" && fixture.calls[0].comm_name->size() == 65535,
           "NULL/empty distinction and exact bounded longest config string");
    call.reset();
    expect(weak.expired() && fixture.count == 1, "trusted quiescent release has no SDK cleanup");
    for (int invalid = 0; invalid != 5; ++invalid) {
        auto bad = arguments(Operation::kInitRankConfig);
        bad.config = full_config();
        if (invalid == 0) bad.config->net_name = std::string(65536, 'x');
        if (invalid == 1) bad.config->comm_name = std::string("a\0b", 3);
        if (invalid == 2) bad.config->declared_bytes = 16;
        if (invalid == 3) {
            bad.config = CudaNcclNativeConfig{};
            bad.config->magic = 1;
        }
        if (invalid == 4) {
            bad.config = CudaNcclNativeConfig{};
            bad.config->declared_bytes = 17;
            bad.config->fields[0] = 256;
        }
        const auto rejected = CudaNcclNativeCall::create(dispatch(), std::move(bad));
        expect(rejected.status == CUDA_ERROR_INVALID_VALUE && !rejected.call,
               "oversized/embedded-NUL/uncaptured scalar and string inputs reject transactionally");
    }
    expect(fixture.count == 1, "malformed configuration creates no native callback");
}

void independent_equal_output_writes() {
    for (const auto operation : {Operation::kGetUniqueId, Operation::kInitRank,
                                 Operation::kInitRankConfig, Operation::kGetAsyncError}) {
        for (int raw = 0; raw != 8; ++raw) {
            for (const bool writes : {false, true}) {
                fixture.reset();
                fixture.no_output = !writes;
                fixture.equal_output = writes;
                fixture.status = static_cast<ncclResult_t>(raw);
                auto request = arguments(operation);
                request.output_contract = CudaNcclNativeOutputContract::kPinnedHostStable;
                auto call = packet(std::move(request));
                expect(call->has_stable_output() && !call->snapshot_after_host_return() &&
                           call->invoke() == CUDA_SUCCESS && call->requested_status() == raw,
                       "pinned stable output remains distinct from actual host return/status");
                const auto observed = call->snapshot_after_host_return();
                expect(observed && observed->has_output && observed->output_written == writes &&
                           observed->communicator_changed == 0 &&
                           observed->async_error_changed == 0 &&
                           observed->unique_id_changed == std::array<std::byte, 16>{},
                       "actual equal-value writes differ from no-write for all statuses and output "
                       "types");
                bool foreign_read{};
                std::thread foreign(
                    [&] { foreign_read = call->snapshot_after_host_return().has_value(); });
                foreign.join();
                expect(!foreign_read,
                       "independent output proof does not widen original-worker authority");
            }
        }
    }
    fixture.reset();
    auto ordinary = packet(arguments(Operation::kGetUniqueId));
    expect(!ordinary->has_stable_output() && ordinary->invoke() == CUDA_SUCCESS &&
               !ordinary->snapshot_after_host_return() &&
               !ordinary->snapshot_after_quiescence()->output_written,
           "default deferred provider cannot acquire the pinned proof by host return");
}

void thread_fork_and_lost_proof() {
    fixture.reset();
    auto call = packet(arguments(Operation::kGetUniqueId));
    CUresult first_result{CUDA_ERROR_UNKNOWN}, replacement_result{CUDA_ERROR_UNKNOWN};
    bool has_replacement_snapshot{};
    std::thread first([&] { first_result = call->invoke(); });
    first.join();
    expect(first_result == CUDA_SUCCESS, "bind original worker");
    std::thread replacement([&] {
        replacement_result = call->invoke();
        has_replacement_snapshot = call->snapshot_after_quiescence().has_value();
    });
    replacement.join();
    expect(replacement_result == CUDA_ERROR_INVALID_CONTEXT && !has_replacement_snapshot,
           "replacement OS thread cannot inherit expired worker incarnation");
    expect(call->requested_status() == 0 && fixture.count == 1,
           "CPU-safe original status survives worker exit without permitting output reads");
    const auto child = ::fork();
    expect(child >= 0, "fork isolated inherited packet guard");
    if (child == 0) {
        const bool valid = call->invoke() == CUDA_ERROR_INVALID_CONTEXT && !call->was_invoked() &&
                           !call->requested_status() && !call->snapshot_after_quiescence();
        std::_Exit(valid ? 0 : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejection occurs before inherited locks/thread-local owner state");
    const auto uncertain = ::fork();
    expect(uncertain >= 0, "fork isolated deliberate lane uncertainty");
    if (uncertain == 0) {
        try {
            fixture.reset();
            fixture.loses_context = true;
            Lane lane;
            const auto packet_call = packet(arguments(Operation::kGetUniqueId));
            const auto ticket =
                lane.submit(packet_call, [packet_call] { return packet_call->invoke(); });
            const auto observed = ticket.observe();
            expect(observed.phase == CudaNativeExecutionCallPhase::kUncertain &&
                       observed.status == CUDA_ERROR_INVALID_CONTEXT && observed.was_invoked &&
                       packet_call->requested_status() == 0 && packet_call->was_invoked() &&
                       lane.lanes.retained_jobs() == 1,
                   "original returned status survives later lane context-proof loss");
            expect(lane.lanes.release(ticket) != CUDA_SUCCESS,
                   "context uncertainty retains packet storage rather than inventing cleanup");
            std::_Exit(0);
        } catch (...) {
            std::_Exit(1);
        }
    }
    expect(::waitpid(uncertain, &status, 0) == uncertain && WIFEXITED(status) &&
               WEXITSTATUS(status) == 0,
           "isolated uncertain lane retains native argument owners through process exit");
}
}  // namespace

int main() {
    try {
        all_eleven_on_stable_lane();
        exact_errors_presence_and_reentry();
        missing_callbacks_and_authority();
        config_prefixes_and_lifetimes();
        independent_equal_output_writes();
        thread_fork_and_lost_proof();
        std::cout << "NCCL native packet: six groups; exact eleven callbacks; "
                     "stable-lane composition; no SDK/GPU qualification\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
