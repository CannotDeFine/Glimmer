#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sched.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
const pid_t kProcess{::getpid()};
constexpr std::array<std::uint64_t, 2> kIncarnation{83, 89};
std::atomic<unsigned int> idle_yields{};
std::atomic<unsigned int> allocation_yields{};
std::atomic<unsigned int> allocation_waits{};
std::atomic<bool> allocation_replied{};
bool should_interrupt_wait{}, did_interrupt_wait{};

// Only the public application's thread installs a watcher. The server uses the
// same real syscalls without instrumentation. No hook injects reply bytes,
// changes readiness, skips a sleep or calls into frontend internals.
struct ReceiveObservation {
    ExecutionOpcode operation{};
    std::uint64_t sequence{};
    std::array<std::byte, 512> frame{};
    std::size_t received_bytes{};
    unsigned int complete_replies{}, positive_receives{}, waiting_yields{}, ready_yields{},
        quick_yields{}, readiness_waits{}, readable_waits{};
    bool is_invalid{};

    void received(const msghdr& message, std::size_t count) noexcept {
        ++positive_receives;
        if (count > frame.size() - received_bytes) {
            is_invalid = true;
            return;
        }
        auto remaining = count;
        for (std::size_t index = 0; index < message.msg_iovlen && remaining; ++index) {
            const auto& item = message.msg_iov[index];
            const auto copied = std::min(remaining, item.iov_len);
            std::memcpy(frame.data() + received_bytes, item.iov_base, copied);
            received_bytes += copied;
            remaining -= copied;
        }
        if (remaining) {
            is_invalid = true;
            return;
        }
        if (received_bytes < kExecutionHeaderBytes) return;
        const auto header = decode_execution_header(std::span{frame}.first(kExecutionHeaderBytes));
        if (!header || header->payload_bytes > frame.size() - kExecutionHeaderBytes ||
            received_bytes > kExecutionHeaderBytes + header->payload_bytes) {
            is_invalid = true;
            return;
        }
        if (received_bytes != kExecutionHeaderBytes + header->payload_bytes) return;
        if (header->kind != ExecutionMessageKind::kReply || header->operation != operation ||
            header->sequence != sequence || !header->session ||
            header->incarnation != kIncarnation) {
            is_invalid = true;
            return;
        }
        ++complete_replies;
        received_bytes = 0;
    }
};
thread_local ReceiveObservation* observation{};
class ObservationScope final {
   public:
    explicit ObservationScope(ReceiveObservation& value) noexcept {
        observation = &value;
    }
    ~ObservationScope() {
        observation = nullptr;
    }
    ObservationScope(const ObservationScope&) = delete;
    ObservationScope& operator=(const ObservationScope&) = delete;
};
struct ObservedCall {
    CUresult status{};
    ReceiveObservation receipt;
};
template <typename Callable>
ObservedCall observe(ExecutionOpcode operation, std::uint64_t sequence, Callable&& call) {
    ReceiveObservation receipt{.operation = operation, .sequence = sequence};
    ObservationScope scope(receipt);
    const auto status = call();
    return {status, receipt};
}
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct ApplicationResult {
    std::array<ObservedCall, 5> calls;
    CUresult released{}, stale_context{};
    CUdeviceptr allocation{}, rejected_allocation{};
    CUcontext rejected_context{};
};
ApplicationResult application() {
    ApplicationResult result;
    CUcontext context{};
    result.calls[0] =
        observe(ExecutionOpcode::kHello, 1, [&] { return cuCtxCreate(&context, 0, 0); });
    expect(result.calls[0].status == CUDA_SUCCESS, "create the ordinary owned context");
    result.calls[1] =
        observe(ExecutionOpcode::kAllocate, 2, [&] { return cuMemAlloc(&result.allocation, 16); });
    expect(result.calls[1].status == CUDA_SUCCESS, "create the owned allocation");
    result.rejected_allocation = result.allocation;
    result.calls[2] = observe(ExecutionOpcode::kAllocate, 3,
                              [&] { return cuMemAlloc(&result.rejected_allocation, 32); });
    result.released = cuMemFree(result.allocation);
    result.calls[3] = observe(ExecutionOpcode::kClose, 5, [&] { return cuCtxDestroy(context); });
    result.stale_context = cuCtxSetCurrent(context);
    result.calls[4] = observe(ExecutionOpcode::kHello, 1,
                              [&] { return cuCtxCreate(&result.rejected_context, 0, 0); });
    return result;
}
void run(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start the actual authenticated fake endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "read the trusted endpoint process identity");
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select the endpoint before the first public frontend operation");
    auto client = std::async(std::launch::async, application);
    std::optional<ExecutionReceivedRequest> held_bootstrap;
    std::optional<ExecutionReceivedRequest> held_allocation;
    std::uint64_t source{}, next_sequence{1};
    unsigned int hellos{}, allocations{}, releases{}, closes{};
    bool has_allocation{};
    const auto reply_to = [&](const ExecutionReceivedRequest& request,
                              const ExecutionWireReply& reply) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = kIncarnation;
        auto bytes = encode_execution_reply(reply);
        expect(bytes && server.reply(request.connection, {header, std::move(*bytes)}),
               "publish the exact real transport reply");
    };
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress the authenticated endpoint");
        if (held_bootstrap && idle_yields.load(std::memory_order_acquire) >= 2) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply_to(*held_bootstrap, reply);
            held_bootstrap.reset();
        }
        if (held_allocation && (allocation_waits.load(std::memory_order_acquire) ||
                                allocation_yields.load(std::memory_order_acquire))) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.id = 7;
            reply.address = 0x100000;
            reply_to(*held_allocation, reply);
            allocation_replied.store(true, std::memory_order_release);
            held_allocation.reset();
        }
        if (auto request = server.take_request()) {
            const auto& header = request->message.header;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            if (header.operation == ExecutionOpcode::kHello) {
                const auto bootstrap = decode_execution_context_bootstrap(request->message.payload);
                expect(bootstrap && bootstrap->purpose == ExecutionContextPurpose::kExplicitUser &&
                           bootstrap->flags == 0 && bootstrap->source_generation == 1 &&
                           header.sequence == 1,
                       "bootstrap is an exact ordinary owned context request");
                if (++hellos == 1) {
                    source = request->connection;
                    next_sequence = 2;
                    held_bootstrap = std::move(request);
                    continue;  // Require a real absent-reply yield before releasing the reply.
                }
                expect(hellos == 2 && closes == 1 && !has_allocation,
                       "the failed bootstrap follows normal context teardown");
                reply.status = CUDA_ERROR_OUT_OF_MEMORY;
                const auto ownership =
                    encode_execution_bootstrap_ownership(ExecutionBootstrapOwnership::kNone);
                expect(ownership.has_value(), "encode known-absent bootstrap ownership");
                reply.data.assign(ownership->begin(), ownership->end());
            } else {
                expect(request->connection == source && header.sequence == next_sequence++,
                       "ordinary requests retain exact source and dense sequence order");
                std::uint64_t value{};
                switch (header.operation) {
                    case ExecutionOpcode::kAllocate:
                        expect(input.read_u64(value) && input.empty() &&
                                   value == (allocations ? 32U : 16U),
                               "allocation payload remains unchanged");
                        if (++allocations == 1) {
                            has_allocation = true;
                            held_allocation = std::move(request);
                            continue;  // Real readiness wait; old sleep permits normal RED cleanup.
                        } else {
                            expect(allocations == 2, "no replay of the rejected allocation");
                            reply.status = CUDA_ERROR_OUT_OF_MEMORY;
                            reply.original_status = reply.status;
                        }
                        break;
                    case ExecutionOpcode::kRelease:
                        expect(
                            has_allocation && input.read_u64(value) && value == 7 && input.empty(),
                            "release the exact successful allocation once");
                        has_allocation = false;
                        ++releases;
                        break;
                    case ExecutionOpcode::kClose:
                        expect(!has_allocation && input.empty(),
                               "terminal Close follows physical fixture cleanup");
                        ++closes;
                        break;
                    default:
                        throw std::runtime_error("unexpected public ready-reply operation");
                }
            }
            reply_to(*request, reply);
        }
        expect(Clock::now() < deadline, "bounded fixture liveness, not a latency threshold");
    }
    const auto result = client.get();
    server.stop();
    // The old frontend reaches every normal release and Close above. Only now
    // test the ready-reply regression, so RED cannot be explained by leaked work.
    expect(!held_bootstrap && !held_allocation && !has_allocation && hellos == 2 &&
               allocations == 2 && releases == 1 && closes == 1 &&
               result.released == CUDA_SUCCESS && result.calls[3].status == CUDA_SUCCESS &&
               result.stale_context == CUDA_ERROR_INVALID_CONTEXT,
           "normal owned cleanup completed before checking progress");
    expect(result.allocation == 0x100000 && result.rejected_allocation == result.allocation &&
               result.calls[2].status == CUDA_ERROR_OUT_OF_MEMORY &&
               result.calls[4].status == CUDA_ERROR_OUT_OF_MEMORY && !result.rejected_context,
           "SDK errors and verified replies surviving transport poison remain intact");
    expect(result.calls[0].receipt.waiting_yields >= 2,
           "an absent exact reply still performs real bounded sleeps");
    for (const auto& call : result.calls)
        expect(!call.receipt.is_invalid && call.receipt.complete_replies == 1 &&
                   call.receipt.positive_receives >= 2 && call.receipt.received_bytes == 0,
               "interposers observed one whole exact header/payload receipt per public call");
    std::cout << "Normal cleanup complete; ready yields: bootstrap="
              << result.calls[0].receipt.ready_yields
              << " success=" << result.calls[1].receipt.ready_yields
              << " error=" << result.calls[2].receipt.ready_yields
              << " close=" << result.calls[3].receipt.ready_yields
              << " rejected-bootstrap=" << result.calls[4].receipt.ready_yields << '\n';
    const auto& allocation = result.calls[1].receipt;
    expect(allocation.quick_yields == 8 && allocation.readiness_waits >= 1,
           "eight quick probes then actual socket readiness");
    if (should_interrupt_wait)
        expect(did_interrupt_wait && allocation.waiting_yields >= 1,
               "one interrupted readiness wait uses normal sleep without losing its exact reply");
    else
        expect(allocation.waiting_yields == 0 && allocation.readable_waits >= 1,
               "native readable readiness, not an unconditional sleep");
    for (const auto& call : result.calls)
        expect(call.receipt.ready_yields == 0,
               "a complete exact reply must be consumed before the next frontend sleep");
}
}  // namespace

extern "C" ssize_t recvmsg(int descriptor, msghdr* message, int flags) {
    const auto count = static_cast<ssize_t>(::syscall(SYS_recvmsg, descriptor, message, flags));
    const auto saved_errno = errno;
    if (count > 0 && observation && ::getpid() == kProcess)
        observation->received(*message, static_cast<std::size_t>(count));
    errno = saved_errno;
    return count;
}
extern "C" int nanosleep(const timespec* requested, timespec* remaining) {
    const auto saved_errno = errno;
    if (observation && ::getpid() == kProcess && requested && requested->tv_sec == 0 &&
        requested->tv_nsec == 50000) {
        if (observation->complete_replies)
            ++observation->ready_yields;
        else {
            ++observation->waiting_yields;
            idle_yields.fetch_add(1, std::memory_order_release);
            if (observation->operation == ExecutionOpcode::kAllocate && observation->sequence == 2)
                allocation_yields.fetch_add(1, std::memory_order_release);
        }
    }
    errno = saved_errno;
    return static_cast<int>(::syscall(SYS_nanosleep, requested, remaining));
}

extern "C" int ppoll(pollfd* descriptors, nfds_t count, const timespec* timeout,
                     const sigset_t* mask) {
    const auto saved_errno = errno;
    const bool is_watched = observation && ::getpid() == kProcess && count == 1 && timeout &&
                            timeout->tv_sec == 0 && timeout->tv_nsec == 50000;
    if (is_watched) {
        ++observation->readiness_waits;
        if (observation->operation == ExecutionOpcode::kAllocate && observation->sequence == 2) {
            if (should_interrupt_wait && !did_interrupt_wait) {
                did_interrupt_wait = true;
                errno = EINTR;
                return -1;  // Test only the interrupted-syscall containment path.
            }
            allocation_waits.fetch_add(1, std::memory_order_release);
            // Hold before the native syscall until the endpoint actually sends
            // its exact reply. Never invent bytes, readiness or a poll result.
            const auto deadline = Clock::now() + std::chrono::seconds{1};
            while (!allocation_replied.load(std::memory_order_acquire) && Clock::now() < deadline)
                (void)::syscall(SYS_sched_yield);
            if (!allocation_replied.load(std::memory_order_acquire)) observation->is_invalid = true;
        }
    }
    errno = saved_errno;
    const auto result =
        static_cast<int>(::syscall(SYS_ppoll, descriptors, count, timeout, mask, _NSIG / 8));
    if (is_watched && result > 0 && (descriptors[0].revents & POLLIN))
        ++observation->readable_waits;
    return result;
}

extern "C" int sched_yield() noexcept {
    if (observation && ::getpid() == kProcess) ++observation->quick_yields;
    return static_cast<int>(::syscall(SYS_sched_yield));
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--interrupt-wait")
        should_interrupt_wait = true;
    else if (argc != 1)
        return EXIT_FAILURE;
    char pattern[] = "/tmp/glimmer-ready-reply-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        run(std::string(directory) + "/service.sock");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
