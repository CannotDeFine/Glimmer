#include "internal/control_request_sequence.h"
#include "internal/remote_control_operation.h"

#include "glimmer/control/unix_socket_server.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/unix_socket_task_execution_server.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <limits>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace {
template <typename T>
decltype(auto) checked(T&& value) noexcept {
    if (!value.has_value()) std::abort();
    return *value;
}

using namespace glimmer::control;
using namespace glimmer::control::internal;
using namespace std::chrono_literals;
using Clock = RemoteControlOperation::Clock;
constexpr NativeEpochExtent kExtent{8, 4, 2, 1'000'000};
constexpr NativeEpochGrant kGrant{{17, 29}, kExtent, 1};
constexpr TaskExecutionIdentity kExecution{17, 29};

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class Descriptor final {
   public:
    explicit Descriptor(int descriptor) : descriptor_(descriptor) {
        require(descriptor >= 0, "owned descriptor must create");
    }
    ~Descriptor() {
        require(::close(descriptor_) == 0, "owned descriptor must close");
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int get() const {
        return descriptor_;
    }

   private:
    int descriptor_;
};

class Directory final {
   public:
    Directory() {
        std::array<char, 48> pattern{};
        constexpr std::string_view prefix = "/tmp/glimmer-remote-operation-XXXXXX";
        std::memcpy(pattern.data(), prefix.data(), prefix.size());
        require(::mkdtemp(pattern.data()) != nullptr, "owned temporary directory must create");
        path = pattern.data();
    }
    ~Directory() {
        require(::rmdir(path.c_str()) == 0, "owned empty directory must release");
    }
    Directory(const Directory&) = delete;
    Directory& operator=(const Directory&) = delete;
    std::string path;
};

class Session final {
   public:
    explicit Session(OwnedControlExchange& endpoint)
        : listener_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        require(path.size() < sizeof(address.sun_path), "owned socket path must fit");
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        if (::bind(listener_.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) !=
            0) {
            std::cerr << "Owned socket bind failed: errno=" << errno << '\n';
            require(false, "socket unavailable is not a passing qualification");
        }
        require(::listen(listener_.get(), 4) == 0, "owned socket must listen");
        auto result = OwnedExchangeColdStatus::kRejected;
        std::thread initialize([&] { result = endpoint.initialize({path, 1000ms}); });
        peer_ = std::make_unique<Descriptor>(
            ::accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC));
        std::string line;
        for (std::size_t index = 0; index < kOwnerControlOpen.size(); ++index) {
            char byte = 0;
            require(::recv(peer_->get(), &byte, 1, 0) == 1, "cold OPEN must arrive");
            line.push_back(byte);
        }
        require(line == kOwnerControlOpen &&
                    ::send(peer_->get(), kOwnerControlReady.data(), kOwnerControlReady.size(),
                           MSG_DONTWAIT | MSG_NOSIGNAL) ==
                        static_cast<ssize_t>(kOwnerControlReady.size()),
                "peer must explicitly negotiate V3");
        require(channel.create_server({17, 29}, SharedControlMode::kReplyHintV3) &&
                    channel.send_offer(peer_->get(), 1000ms),
                "real sealed V3 offer must send");
        initialize.join();
        require(result == OwnedExchangeColdStatus::kReady,
                "owned endpoint must bind before publication");
    }
    ~Session() {
        require(::unlink(path.c_str()) == 0, "only owned socket inode must remove");
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    std::string receive() {
        std::string line;
        require(channel.receive_request(peer_->get(), 1000, line) == SharedControlReadStatus::kLine,
                "one actual request must arrive");
        return line;
    }
    void reply(std::string_view line) {
        require(channel.publish_response_with_hint(peer_->get(), line),
                "actual reply must precede hint");
    }
    Directory directory;
    const std::string path{directory.path + "/control.sock"};
    SharedControlChannel channel;

   private:
    Descriptor listener_;
    std::unique_ptr<Descriptor> peer_;
};

RemoteControlResult result(RemoteControlOperation& operation) {
    auto value = operation.take_result();
    require(value.has_value() && !operation.take_result(),
            "typed result may be consumed exactly once");
    return checked(value);
}

void start(RemoteControlOperation& operation, OwnedControlExchange& endpoint,
           Clock::time_point now) {
    require(!operation.take_result() &&
                operation.begin(endpoint, now) == RemoteControlStep::kPending &&
                operation.was_published() &&
                operation.begin(endpoint, now) == RemoteControlStep::kRejected,
            "operation must publish exactly once and not yield an unready result");
    const auto wait = operation.wait_state();
    require(wait && wait->deadline == now + 1000ms, "actual request retains its fixed deadline");
    for (std::size_t attempt = 0; attempt < 8; ++attempt)
        require(operation.poll(now) == RemoteControlStep::kPending && !operation.take_result() &&
                    checked(operation.wait_state()).deadline == checked(wait).deadline,
                "unready polling cannot grant, acknowledge, wait or renew expiry");
}

TaskExecutionProtocolResponse response(TaskExecutionProtocolResponseKind kind) {
    TaskExecutionProtocolResponse value{};
    value.kind = kind;
    if (kind != TaskExecutionProtocolResponseKind::kError &&
        kind != TaskExecutionProtocolResponseKind::kEmpty)
        value.task_id = kExecution.task_id;
    if (kind == TaskExecutionProtocolResponseKind::kExecutionLease ||
        kind == TaskExecutionProtocolResponseKind::kExecutionHandoff ||
        kind == TaskExecutionProtocolResponseKind::kRetreat)
        value.generation = kExecution.generation;
    value.tenant_id = "default";
    value.memory_bytes = 1;
    value.weight = 1;
    value.work_units = 1;
    if (kind == TaskExecutionProtocolResponseKind::kExecutionHandoff)
        value.completed_execution = kExecution;
    return value;
}

RemoteControlResult scripted(RemoteCooperativeCommand command,
                             const TaskExecutionProtocolResponse& reply,
                             RemoteControlStep expected = RemoteControlStep::kFinished) {
    ServiceIncarnationBinding binding;
    OwnedControlExchange endpoint{binding};
    Session session{endpoint};
    auto operation = RemoteControlOperation::prepare(command, {});
    require(operation != nullptr, "cooperative operation must prepare");
    const auto now = Clock::now();
    start(*operation, endpoint, now);
    const auto request = parse_task_execution_protocol_request(session.receive());
    require(request.parsed() && checked(request.request).operation == command.operation,
            "canonical command kind must survive preparation");
    if (command.execution.task_id != 0)
        require(checked(request.request).task_id == command.execution.task_id,
                "known task must survive exact wire encoding");
    const auto wire = format_task_execution_protocol_response(reply);
    require(wire.has_value(), "scripted reply must use valid canonical framing/schema");
    session.reply(checked(wire));
    require(operation->poll(now) == expected && !operation->wait_state(),
            "actual reply must be validated before finishing or quarantining");
    return result(*operation);
}

void test_cooperative_replies() {
    const auto accepted = response(TaskExecutionProtocolResponseKind::kAccepted);
    require(std::get<std::optional<glimmer::core::TaskId>>(
                scripted({TaskExecutionProtocolOperation::kSubmit}, accepted)) == 17,
            "submit must decode actual accepted identity");
    auto pending = std::get<CooperativeSubmission>(
        scripted({TaskExecutionProtocolOperation::kExecutionAcquire}, accepted));
    require(pending.task_id == 17 && pending.claim.status == CooperativeClaimStatus::kPending,
            "queued acquisition is owned pending, not authorization");
    const auto lease = response(TaskExecutionProtocolResponseKind::kExecutionLease);
    auto acquired = std::get<CooperativeSubmission>(
        scripted({TaskExecutionProtocolOperation::kExecutionAcquire}, lease));
    require(acquired.task_id == 17 && acquired.claim.status == CooperativeClaimStatus::kGranted &&
                acquired.claim.execution.generation == 29,
            "actual lease must carry its generation");
    const auto claim = std::get<CooperativeClaim>(
        scripted({TaskExecutionProtocolOperation::kExecutionClaim, kExecution}, lease));
    require(claim.status == CooperativeClaimStatus::kGranted && claim.execution.generation == 29,
            "claim must match requested task and nonzero generation");
    require(std::get<CooperativeClaim>(
                scripted({TaskExecutionProtocolOperation::kExecutionClaim, kExecution},
                         response(TaskExecutionProtocolResponseKind::kEmpty)))
                    .status == CooperativeClaimStatus::kPending,
            "empty is pending, not completion");
    auto retreat = response(TaskExecutionProtocolResponseKind::kRetreat);
    retreat.should_retreat = true;
    require(std::get<std::optional<bool>>(
                scripted({TaskExecutionProtocolOperation::kRetreat, kExecution}, retreat)) == true,
            "retreat must validate actual task/generation");
    for (const auto& [operation, state] :
         {std::pair{TaskExecutionProtocolOperation::kExecutionHeartbeat,
                    TaskProtocolState::kRunning},
          std::pair{TaskExecutionProtocolOperation::kExecutionHeartbeat,
                    TaskProtocolState::kQueued},
          std::pair{TaskExecutionProtocolOperation::kSuspend, TaskProtocolState::kQueued},
          std::pair{TaskExecutionProtocolOperation::kExecutionComplete,
                    TaskProtocolState::kCompleted},
          std::pair{TaskExecutionProtocolOperation::kExecutionFail, TaskProtocolState::kFailed},
          std::pair{TaskExecutionProtocolOperation::kExecutionCancel,
                    TaskProtocolState::kCancelled},
          std::pair{TaskExecutionProtocolOperation::kCancel, TaskProtocolState::kCancelled}}) {
        auto ack = response(TaskExecutionProtocolResponseKind::kState);
        ack.state = state;
        require(std::get<bool>(scripted({operation, kExecution}, ack)),
                "only canonical matching STATE acknowledges a report");
    }
    auto handoff = response(TaskExecutionProtocolResponseKind::kExecutionHandoff);
    handoff.task_id = 18;
    handoff.generation = 30;
    const auto completed = std::get<CooperativeHandoff>(
        scripted({TaskExecutionProtocolOperation::kExecutionCompleteAcquire, kExecution}, handoff));
    require(completed.is_completed && completed.submission.task_id == 18 &&
                completed.submission.claim.execution.generation == 30,
            "combined response retains actual old ACK and distinct new authorization");
    auto wrong = lease;
    wrong.task_id = 18;
    require(std::get<CooperativeClaim>(
                scripted({TaskExecutionProtocolOperation::kExecutionClaim, kExecution}, wrong,
                         RemoteControlStep::kUncertain))
                    .status == CooperativeClaimStatus::kFailed,
            "foreign claim cannot fund requested task");
    retreat.generation = 30;
    require(!std::get<std::optional<bool>>(
                scripted({TaskExecutionProtocolOperation::kRetreat, kExecution}, retreat,
                         RemoteControlStep::kUncertain)),
            "stale generation cannot acknowledge retreat");
    handoff.completed_execution.generation = 30;
    require(!std::get<CooperativeHandoff>(
                 scripted({TaskExecutionProtocolOperation::kExecutionCompleteAcquire, kExecution},
                          handoff, RemoteControlStep::kUncertain))
                 .is_completed,
            "wrong old generation cannot acknowledge handoff");
    auto error = response(TaskExecutionProtocolResponseKind::kError);
    error.error = TaskProtocolErrorCode::kInvalidRequest;
    require(!std::get<bool>(
                scripted({TaskExecutionProtocolOperation::kExecutionComplete, kExecution}, error)),
            "explicit service rejection is failed result, not ACK");
    error.error = TaskProtocolErrorCode::kInternalError;
    require(
        !std::get<bool>(scripted({TaskExecutionProtocolOperation::kExecutionComplete, kExecution},
                                 error, RemoteControlStep::kUncertain)),
        "internal error cannot prove terminal ownership was retired");
}

RemoteControlResult scripted(RemoteNativeCommand command, NativeEpochResponse reply,
                             RemoteControlStep expected = RemoteControlStep::kFinished,
                             bool has_wrong_sequence = false) {
    ServiceIncarnationBinding binding;
    OwnedControlExchange endpoint{binding};
    Session session{endpoint};
    auto operation = RemoteControlOperation::prepare(command, {}, kExtent);
    require(operation != nullptr, "native operation must prepare");
    const auto now = Clock::now();
    start(*operation, endpoint, now);
    const auto request = parse_native_epoch_request(session.receive());
    require(request && request->operation == command.operation && request->request_sequence != 0,
            "each native command requires fresh correlation and exact operation");
    reply.request_sequence = checked(request).request_sequence + (has_wrong_sequence ? 1 : 0);
    const auto wire = format_native_epoch_response(reply);
    require(wire.has_value(), "native scripted reply must be valid schema");
    session.reply(checked(wire));
    require(operation->poll(now) == expected, "native response must match all required fields");
    return result(*operation);
}

void test_native_replies() {
    const RemoteNativeCommand acquire{.operation = NativeEpochOperation::kAcquire,
                                      .settings = {.extent = kExtent}};
    NativeEpochResponse reply{};
    reply.kind = NativeEpochResponseKind::kGranted;
    reply.grant = kGrant;
    const auto expired = std::get<NativeEpochClaim>(scripted(acquire, reply));
    require(expired.status == NativeEpochClaimStatus::kGranted &&
                expired.grant.acceptance_deadline_nanoseconds == 1,
            "expired actual grant retains ownership/deadline, not new issue eligibility");
    require(
        std::get<NativeEpochClaim>(scripted(acquire, reply, RemoteControlStep::kUncertain, true))
                .status == NativeEpochClaimStatus::kUncertain,
        "wrong native sequence cannot grant");
    reply.grant.extent.max_calls = 9;
    require(std::get<NativeEpochClaim>(scripted(acquire, reply, RemoteControlStep::kUncertain))
                    .status == NativeEpochClaimStatus::kUncertain,
            "wrong native extent cannot grant");
    reply.grant = kGrant;
    reply.grant.acceptance_deadline_nanoseconds = std::numeric_limits<std::uint64_t>::max();
    require(std::get<NativeEpochClaim>(scripted(acquire, reply, RemoteControlStep::kUncertain))
                    .status == NativeEpochClaimStatus::kUncertain,
            "excessive deadline cannot widen authorization");
    reply.kind = NativeEpochResponseKind::kPending;
    reply.grant = {.execution = {17, 0}, .extent = kExtent};
    const auto pending = std::get<NativeEpochClaim>(scripted(acquire, reply));
    require(
        pending.status == NativeEpochClaimStatus::kPending && pending.grant.execution.task_id == 17,
        "pending acquisition retains exact extent and identity");
    const RemoteNativeCommand claim{.operation = NativeEpochOperation::kClaim,
                                    .grant = pending.grant};
    require(std::get<NativeEpochClaim>(scripted(claim, reply)).status ==
                NativeEpochClaimStatus::kPending,
            "pending claim may remain pending");
    reply.grant.execution.task_id = 18;
    require(std::get<NativeEpochClaim>(scripted(claim, reply, RemoteControlStep::kUncertain))
                    .grant.execution.task_id == 17,
            "wrong claim keeps original pending ownership");
    reply.kind = NativeEpochResponseKind::kRetreat;
    reply.operation = NativeEpochOperation::kRetreat;
    reply.grant = kGrant;
    reply.should_retreat = true;
    require(std::get<std::optional<bool>>(
                scripted({NativeEpochOperation::kRetreat, kGrant}, reply)) == true,
            "native retreat requires actual matching generation");
    for (const auto operation :
         {NativeEpochOperation::kHeartbeat, NativeEpochOperation::kComplete,
          NativeEpochOperation::kFail, NativeEpochOperation::kCancelPending}) {
        const auto grant =
            operation == NativeEpochOperation::kCancelPending ? pending.grant : kGrant;
        const std::uint32_t count =
            operation == NativeEpochOperation::kComplete || operation == NativeEpochOperation::kFail
                ? 2
                : 0;
        reply.kind = NativeEpochResponseKind::kAcknowledged;
        reply.operation = operation;
        reply.grant = grant;
        reply.issued_calls = count;
        require(std::get<bool>(scripted(
                    {.operation = operation, .grant = grant, .issued_calls = count}, reply)),
                "native reports require actual matching operation/task/generation/count");
        reply.grant.execution.task_id = 18;
        require(!std::get<bool>(
                    scripted({.operation = operation, .grant = grant, .issued_calls = count}, reply,
                             RemoteControlStep::kUncertain)),
                "foreign native ACK cannot release ownership");
    }
    reply.kind = NativeEpochResponseKind::kError;
    reply.error = TaskProtocolErrorCode::kQuotaExceeded;
    require(std::get<NativeEpochClaim>(scripted(acquire, reply, RemoteControlStep::kUncertain))
                    .status == NativeEpochClaimStatus::kUncertain,
            "native error cannot prove admission was not published");
    require(std::get<NativeEpochClaim>(scripted(claim, reply, RemoteControlStep::kUncertain))
                    .grant.execution.task_id == 17,
            "failed pending claim cannot erase known ownership");
}

void test_fallback_and_uncertainty() {
    for (const bool has_lost_primary : {false, true}) {
        ServiceIncarnationBinding binding;
        OwnedControlExchange endpoint{binding};
        Session session{endpoint};
        LaunchGateOptions options;
        options.tenant_id = std::string(96, 'a');
        options.tenant_weight = std::numeric_limits<std::uint32_t>::max();
        options.task_priority = std::numeric_limits<std::uint32_t>::max();
        RemoteCooperativeCommand command{
            TaskExecutionProtocolOperation::kExecutionCompleteAcquire,
            {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()},
            {.work_units = 0}};
        auto operation = RemoteControlOperation::prepare(command, options);
        require(operation != nullptr, "fallback must prepare before publication");
        const auto now = Clock::now();
        start(*operation, endpoint, now);
        const auto first = parse_task_execution_protocol_request(session.receive());
        require(first.parsed() && checked(first.request).operation ==
                                      TaskExecutionProtocolOperation::kExecutionComplete,
                "only local combined encoding rejection permits independent primary report");
        std::string unexpected;
        require(
            session.channel.receive_request(-1, 1, unexpected) == SharedControlReadStatus::kError,
            "no unrelated borrowed socket may consume a successor");
        for (std::size_t index = 0; index < 16; ++index)
            require(operation->poll(now) == RemoteControlStep::kPending &&
                        !operation->completion_acknowledged(),
                    "withheld actual primary ACK cannot publish successor or claim completion");
        if (has_lost_primary) {
            require(operation->poll(now + 1000ms) == RemoteControlStep::kUncertain &&
                        !operation->completion_acknowledged() &&
                        !std::get<CooperativeHandoff>(result(*operation)).is_completed,
                    "lost primary acknowledgement cannot fund fallback or release ownership");
            continue;
        }
        auto ack = response(TaskExecutionProtocolResponseKind::kState);
        ack.task_id = command.execution.task_id;
        ack.state = TaskProtocolState::kCompleted;
        const auto encoded = format_task_execution_protocol_response(ack);
        require(encoded.has_value(), "actual primary ACK must encode");
        session.reply(checked(encoded));
        const auto next = operation->poll(now);
        require(operation->completion_acknowledged(), "actual old ACK is retained independently");
        require(next == RemoteControlStep::kFinished,
                "invalid new request publishes no secondary after valid primary acknowledgement");
        const auto handoff = std::get<CooperativeHandoff>(result(*operation));
        require(handoff.is_completed && handoff.submission.task_id == 0,
                "secondary failure retains real primary ACK but no guessed new grant");
    }
    ServiceIncarnationBinding binding;
    OwnedControlExchange endpoint{binding};
    Session session{endpoint};
    auto operation =
        RemoteControlOperation::prepare({TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    const auto now = Clock::now();
    start(*operation, endpoint, now);
    static_cast<void>(session.receive());
    session.reply("WRONG\n");
    require(operation->poll(now) == RemoteControlStep::kUncertain &&
                operation->begin(endpoint, now) == RemoteControlStep::kRejected,
            "malformed published response cannot replay or repair operation");
    auto later =
        RemoteControlOperation::prepare({TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    require(later && later->begin(endpoint, now) == RemoteControlStep::kFinished &&
                !later->was_published(),
            "quarantined domain reply retires idle transport before another implicit publication");
}

void test_preparation_and_fork() {
    ServiceIncarnationBinding binding;
    OwnedControlExchange endpoint{binding};
    auto invalid = RemoteControlOperation::prepare(
        {TaskExecutionProtocolOperation::kExecutionComplete, {}}, {});
    require(invalid && invalid->begin(endpoint) == RemoteControlStep::kFinished &&
                !invalid->was_published() && !std::get<bool>(result(*invalid)),
            "invalid execution rejects before publication");
    require(!RemoteControlOperation::prepare({TaskExecutionProtocolOperation::kStats}, {}),
            "non-cooperative commands cannot enter this adapter");
    auto denied = RemoteControlOperation::prepare(
        RemoteNativeCommand{.operation = NativeEpochOperation::kAcquire,
                            .settings = {.extent = kExtent}},
        {}, std::nullopt);
    require(
        denied && denied->begin(endpoint) == RemoteControlStep::kFinished &&
            !denied->was_published() &&
            std::get<NativeEpochClaim>(result(*denied)).status == NativeEpochClaimStatus::kRejected,
        "disabled policy cannot publish native admission");
    Session session{endpoint};
    auto operation =
        RemoteControlOperation::prepare({TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    const auto now = Clock::now();
    start(*operation, endpoint, now);
    static_cast<void>(session.receive());
    const pid_t child = ::fork();
    require(child >= 0, "owned fork child must create");
    if (child == 0) {
        const bool is_rejected = operation->begin(endpoint, now) == RemoteControlStep::kRejected &&
                                 operation->poll(now) == RemoteControlStep::kRejected &&
                                 !operation->wait_state() && !operation->take_result();
        operation.reset();
        ::_exit(is_rejected ? 0 : 1);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "owned fork rejection child must terminate and be reaped");
    session.reply("GLIMMER_TASK_EXEC_V1 EXEC_LEASE 17 default 1 1 1 29\n");
    require(operation->poll(now) == RemoteControlStep::kFinished,
            "fork rejection cannot consume or abandon parent operation");
    const pid_t ready_child = ::fork();
    require(ready_child >= 0, "ready-result fork child must create");
    if (ready_child == 0) ::_exit(operation->take_result() ? 1 : 0);
    require(::waitpid(ready_child, &status, 0) == ready_child && WIFEXITED(status) &&
                WEXITSTATUS(status) == 0 &&
                std::get<CooperativeSubmission>(result(*operation)).claim.status ==
                    CooperativeClaimStatus::kGranted,
            "fork cannot inherit cached positive authorization and parent still consumes once");
    auto pending =
        RemoteControlOperation::prepare({TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    start(*pending, endpoint, now);
    static_cast<void>(session.receive());
    pending.reset();
    auto successor =
        RemoteControlOperation::prepare({TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    require(successor->begin(endpoint, now) == RemoteControlStep::kFinished &&
                !successor->was_published(),
            "pending destruction quarantines transport without reporting physical stop");
}

void test_native_replay_timeout_and_bounds() {
    ServiceIncarnationBinding binding;
    OwnedControlExchange endpoint{binding};
    Session session{endpoint};
    const RemoteNativeCommand acquire{.operation = NativeEpochOperation::kAcquire,
                                      .settings = {.extent = kExtent}};
    auto first = RemoteControlOperation::prepare(acquire, {}, kExtent);
    const auto now = Clock::now();
    start(*first, endpoint, now);
    const auto first_request = parse_native_epoch_request(session.receive());
    require(first_request.has_value(), "first actual native request must parse");
    NativeEpochResponse granted{};
    granted.kind = NativeEpochResponseKind::kGranted;
    granted.grant = kGrant;
    granted.grant.acceptance_deadline_nanoseconds =
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
                .count()) +
        100'000'000;
    granted.request_sequence = checked(first_request).request_sequence;
    const auto old_wire = format_native_epoch_response(granted);
    require(old_wire.has_value(), "first actual granted frame must encode");
    session.reply(checked(old_wire));
    require(first->poll(now) == RemoteControlStep::kFinished,
            "first native response must validate");
    const auto owned = std::get<NativeEpochClaim>(result(*first));
    require(owned.status == NativeEpochClaimStatus::kGranted,
            "first native grant must retain actual ownership");
    auto finish = RemoteControlOperation::prepare(
        RemoteNativeCommand{
            .operation = NativeEpochOperation::kComplete, .grant = owned.grant, .issued_calls = 1},
        {}, kExtent);
    start(*finish, endpoint, now);
    const auto finish_request = parse_native_epoch_request(session.receive());
    require(finish_request &&
                finish_request->request_sequence != checked(first_request).request_sequence,
            "completion must use a distinct fresh process tag");
    NativeEpochResponse ack{};
    ack.kind = NativeEpochResponseKind::kAcknowledged;
    ack.operation = NativeEpochOperation::kComplete;
    ack.grant = owned.grant;
    ack.issued_calls = 1;
    ack.request_sequence = checked(finish_request).request_sequence;
    const auto ack_wire = format_native_epoch_response(ack);
    require(ack_wire.has_value(), "actual synthetic-completion acknowledgement must encode");
    session.reply(checked(ack_wire));
    require(finish->poll(now) == RemoteControlStep::kFinished && std::get<bool>(result(*finish)),
            "actual CPU-declared completion must independently acknowledge first grant");
    auto replayed = RemoteControlOperation::prepare(acquire, {}, kExtent);
    start(*replayed, endpoint, now);
    const auto later = parse_native_epoch_request(session.receive());
    require(later && later->request_sequence != checked(first_request).request_sequence &&
                later->request_sequence != checked(finish_request).request_sequence,
            "new acquisition cannot reuse prior tag");
    session.reply(checked(old_wire));
    require(replayed->poll(now) == RemoteControlStep::kUncertain &&
                std::get<NativeEpochClaim>(result(*replayed)).status ==
                    NativeEpochClaimStatus::kUncertain,
            "byte-identical completed prior grant cannot fund a later acquisition");
    {
        ServiceIncarnationBinding logical;
        OwnedControlExchange pending_endpoint{logical};
        Session peer{pending_endpoint};
        NativeEpochGrant pending{.execution = {17, 0}, .extent = kExtent};
        auto claim = RemoteControlOperation::prepare(
            RemoteNativeCommand{NativeEpochOperation::kClaim, pending}, {}, kExtent);
        start(*claim, pending_endpoint, now);
        static_cast<void>(peer.receive());
        require(claim->poll(now + 1000ms) == RemoteControlStep::kUncertain,
                "unready pending native claim expires without replay");
        const auto retained = std::get<NativeEpochClaim>(result(*claim));
        require(retained.status == NativeEpochClaimStatus::kUncertain &&
                    retained.grant.execution.task_id == 17 && retained.grant.extent == kExtent,
                "claim timeout cannot erase known pending identity/extent");
    }
    {
        ServiceIncarnationBinding logical;
        OwnedControlExchange uninitialized{logical};
        auto excessive = acquire;
        excessive.settings.extent.max_calls = 9;
        auto denied = RemoteControlOperation::prepare(excessive, {}, kExtent);
        require(denied && denied->begin(uninitialized) == RemoteControlStep::kFinished &&
                    !denied->was_published() &&
                    std::get<NativeEpochClaim>(result(*denied)).status ==
                        NativeEpochClaimStatus::kRejected,
                "excessive policy cannot be widened or published");
        auto report = RemoteControlOperation::prepare(
            RemoteNativeCommand{
                .operation = NativeEpochOperation::kComplete, .grant = kGrant, .issued_calls = 9},
            {}, kExtent);
        require(report && report->begin(uninitialized) == RemoteControlStep::kFinished &&
                    !report->was_published() && !std::get<bool>(result(*report)),
                "excessive terminal count cannot be acknowledged or published");
    }
    ack.grant = kGrant;
    ack.issued_calls = 3;
    require(
        !std::get<bool>(scripted(
            RemoteNativeCommand{
                .operation = NativeEpochOperation::kComplete, .grant = kGrant, .issued_calls = 2},
            ack, RemoteControlStep::kUncertain)),
        "wrong actual issued count cannot acknowledge terminal report");
    ack.operation = NativeEpochOperation::kFail;
    ack.issued_calls = 2;
    require(
        !std::get<bool>(scripted(
            RemoteNativeCommand{
                .operation = NativeEpochOperation::kComplete, .grant = kGrant, .issued_calls = 2},
            ack, RemoteControlStep::kUncertain)),
        "wrong actual operation cannot acknowledge terminal report");
}

bool registered(void*, glimmer::core::TaskId) noexcept {
    return true;
}
void rollback(void*, glimmer::core::TaskId) noexcept {}

class Service final {
   public:
    Service()
        : endpoint_(std::make_unique<TaskExecutionAdmissionService>(
                        1024,
                        glimmer::core::SchedulerOptions{
                            .scheduling_policy = glimmer::core::SchedulingPolicy::kPriority},
                        5000ms, true, nullptr, nullptr, nullptr, nullptr, kExtent),
                    registered, rollback, nullptr),
          server_(endpoint_,
                  {.socket_path = path, .allowed_uid = std::nullopt, .io_timeout_ms = 1000}) {
        require(server_.start(), "actual process-bound service must start");
        acceptor_ = std::jthread([this](const std::stop_token& stop) {
            while (!stop.stop_requested()) {
                const auto status = server_.serve_one_for(10);
                require(status != UnixSocketServeStatus::kError || stop.stop_requested(),
                        "owned acceptor cannot fail outside explicit teardown");
            }
        });
    }
    ~Service() {
        require(endpoint_.begin_shutdown(), "stop new work before service teardown");
        acceptor_.request_stop();
        acceptor_.join();
        server_.stop();
    }
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;
    Directory directory;
    const std::string path{directory.path + "/control.sock"};
    std::optional<TaskProtocolState> task_state(glimmer::core::TaskId id) {
        const auto line =
            endpoint_.handle("GLIMMER_TASK_EXEC_V1 QUERY " + std::to_string(id) + "\n");
        const auto parsed = line ? parse_task_execution_protocol_response(*line)
                                 : TaskExecutionProtocolResponseParseResult{};
        return parsed.parsed() ? std::optional{checked(parsed.response).state} : std::nullopt;
    }
    std::optional<TaskExecutionLeaseStatus> status() const {
        return endpoint_.lease_status();
    }

   private:
    TaskExecutionEndpoint endpoint_;
    UnixSocketTaskExecutionServer server_;
    std::jthread acceptor_;
};

RemoteControlResult run(std::unique_ptr<RemoteControlOperation> operation,
                        OwnedControlExchange& endpoint) {
    require(operation && operation->begin(endpoint) == RemoteControlStep::kPending,
            "real operation must start");
    auto state = RemoteControlStep::kPending;
    const auto deadline = Clock::now() + 2s;
    while (state == RemoteControlStep::kPending && Clock::now() < deadline) {
        state = operation->poll();
        if (state == RemoteControlStep::kPending) {
            const auto wait = operation->wait_state();
            require(wait.has_value(), "pending real operation retains wait state");
            pollfd reply{.fd = checked(wait).descriptor, .events = POLLIN, .revents = 0};
            require(::poll(&reply, 1, 1000) > 0,
                    "actual service response must wake wait descriptor");
        }
    }
    require(state == RemoteControlStep::kFinished, "actual authenticated operation must finish");
    return result(*operation);
}

void test_authenticated_service_and_shared_sequence() {
    Service service;
    ServiceIncarnationBinding binding;
    OwnedControlExchange endpoint{binding};
    require(endpoint.initialize({service.path, 1000ms}) == OwnedExchangeColdStatus::kReady,
            "real service must negotiate before domain admission");
    const auto cooperative = [&](RemoteCooperativeCommand command) {
        return run(RemoteControlOperation::prepare(command, {}), endpoint);
    };
    const auto native = [&](RemoteNativeCommand command) {
        return run(RemoteControlOperation::prepare(command, {}, kExtent), endpoint);
    };
    const auto submitted = std::get<std::optional<glimmer::core::TaskId>>(
        cooperative({TaskExecutionProtocolOperation::kSubmit}));
    require(submitted.has_value(), "real submit must retain accepted task");
    const auto claim = std::get<CooperativeClaim>(
        cooperative({TaskExecutionProtocolOperation::kExecutionClaim, {checked(submitted), 0}}));
    require(claim.status == CooperativeClaimStatus::kGranted, "real claim must obtain generation");
    require(std::get<std::optional<bool>>(cooperative(
                {TaskExecutionProtocolOperation::kRetreat, claim.execution})) == false &&
                std::get<bool>(cooperative(
                    {TaskExecutionProtocolOperation::kExecutionHeartbeat, claim.execution})) &&
                std::get<bool>(
                    cooperative({TaskExecutionProtocolOperation::kSuspend, claim.execution})),
            "real retreat/heartbeat/suspend must use actual reports");
    const auto resumed = std::get<CooperativeClaim>(
        cooperative({TaskExecutionProtocolOperation::kExecutionClaim, claim.execution}));
    require(resumed.status == CooperativeClaimStatus::kGranted &&
                resumed.execution.generation != claim.execution.generation &&
                std::get<bool>(cooperative(
                    {TaskExecutionProtocolOperation::kExecutionFail, resumed.execution})),
            "resumed execution requires a fresh generation before synthetic CPU fail");
    const auto ordinary = std::get<CooperativeSubmission>(
        cooperative({TaskExecutionProtocolOperation::kExecutionAcquire}));
    require(ordinary.claim.status == CooperativeClaimStatus::kGranted,
            "real acquisition must grant");
    const auto handoff = std::get<CooperativeHandoff>(cooperative(
        {TaskExecutionProtocolOperation::kExecutionCompleteAcquire, ordinary.claim.execution}));
    require(handoff.is_completed &&
                handoff.submission.claim.status == CooperativeClaimStatus::kGranted &&
                handoff.submission.task_id != ordinary.task_id,
            "actual combined old ACK must return distinct successor grant");
    require(!std::get<bool>(cooperative({TaskExecutionProtocolOperation::kExecutionCancel,
                                         handoff.submission.claim.execution})) &&
                service.task_state(handoff.submission.task_id) == TaskProtocolState::kRunning,
            "running execution cannot use continuation cancellation or release capacity");
    require(std::get<bool>(cooperative(
                {TaskExecutionProtocolOperation::kSuspend, handoff.submission.claim.execution})) &&
                std::get<bool>(cooperative({TaskExecutionProtocolOperation::kExecutionCancel,
                                            handoff.submission.claim.execution})),
            "stopped cancellation requires actual queued continuation acknowledgement");
    const auto queued = std::get<std::optional<glimmer::core::TaskId>>(
        cooperative({TaskExecutionProtocolOperation::kSubmit}));
    require(queued && std::get<bool>(
                          cooperative({TaskExecutionProtocolOperation::kCancel, {*queued, 0}})),
            "only never-dispatched task uses pending cancellation");
    const RemoteNativeCommand acquire{.operation = NativeEpochOperation::kAcquire,
                                      .settings = {.extent = kExtent}};
    auto epoch = std::get<NativeEpochClaim>(native(acquire));
    require(epoch.status == NativeEpochClaimStatus::kGranted &&
                std::get<std::optional<bool>>(
                    native({NativeEpochOperation::kRetreat, epoch.grant})) == false &&
                std::get<bool>(native({NativeEpochOperation::kHeartbeat, epoch.grant})) &&
                std::get<bool>(native({.operation = NativeEpochOperation::kComplete,
                                       .grant = epoch.grant,
                                       .issued_calls = 2})),
            "real native grant/probe/report must retain exact extent/identity/count");
    epoch = std::get<NativeEpochClaim>(native(acquire));
    require(epoch.status == NativeEpochClaimStatus::kGranted &&
                std::get<bool>(native({.operation = NativeEpochOperation::kFail,
                                       .grant = epoch.grant,
                                       .issued_calls = 2})),
            "real native fail acknowledges only its own CPU-declared completion");
    const auto blocker = std::get<CooperativeSubmission>(
        cooperative({TaskExecutionProtocolOperation::kExecutionAcquire}));
    epoch = std::get<NativeEpochClaim>(native(acquire));
    require(epoch.status == NativeEpochClaimStatus::kPending &&
                std::get<NativeEpochClaim>(native({NativeEpochOperation::kClaim, epoch.grant}))
                        .status == NativeEpochClaimStatus::kPending &&
                std::get<bool>(native({NativeEpochOperation::kCancelPending, epoch.grant})) &&
                std::get<bool>(cooperative(
                    {TaskExecutionProtocolOperation::kExecutionComplete, blocker.claim.execution})),
            "queued epoch cannot acquire while ordinary running slot remains owned");
    LaunchGateOptions options;
    options.control_socket = service.path;
    const auto before = next_epoch_request_sequence();
    NativeEpochClaim original;
    bool is_completed = false;
    std::thread reference([&] {
        CooperativeLaunchGate synchronous{options, kExtent};
        original = synchronous.acquire_native_epoch({.extent = kExtent});
        is_completed = synchronous.finish_native_epoch(original.grant, 0, false);
    });
    reference.join();
    require(original.status == NativeEpochClaimStatus::kGranted && is_completed,
            "unchanged synchronous adapter must retain its actual native protocol");
    const auto after = next_epoch_request_sequence();
    require(after == before + 3,
            "old and new adapters must use one process-lifetime sequence, not duplicate counters");
    const auto service_status = service.status();
    require(service_status && service_status->running_tasks == 0 &&
                service_status->queued_tasks == 0 && endpoint.close_idle(),
            "all synthetic CPU-declared work and endpoint resources must drain");
}

std::size_t descriptor_count() {
    DIR* directory = ::opendir("/proc/self/fd");
    require(directory != nullptr, "descriptor inventory must open");
    std::size_t count = 0;
    errno = 0;
    while (const auto* entry = ::readdir(directory))
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    require(errno == 0 && ::closedir(directory) == 0, "descriptor inventory must close");
    return count;
}

}  // namespace

int main() {
    const auto descriptors = descriptor_count();
    test_cooperative_replies();
    test_native_replies();
    test_fallback_and_uncertainty();
    test_preparation_and_fork();
    test_native_replay_timeout_and_bounds();
    test_authenticated_service_and_shared_sequence();
    require(descriptor_count() == descriptors,
            "all accepted/rejected/error/fork CPU descriptors must release");
}
