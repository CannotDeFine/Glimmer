#include "execution/copy_audit.h"
#include "execution/rpc_metrics.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <sys/wait.h>

namespace {
std::array<char, 16384> output;
std::size_t output_size{};
std::atomic<std::uint64_t> clock_calls{}, cpu_clock_calls{}, ticks{};
enum class ClockMode { kNormal, kMissing, kOverflow };
ClockMode clock_mode{};
bool should_truncate{};
void expect(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
void reset() {
    output_size = 0;
    clock_calls = 0;
    cpu_clock_calls = 0;
    ticks = 100;
    clock_mode = ClockMode::kNormal;
    should_truncate = false;
}
std::string_view captured() {
    return {output.data(), output_size};
}
}  // namespace

// NOLINTBEGIN(bugprone-reserved-identifier,readability-identifier-naming): GNU ld wrapping ABI.
extern "C" int __real_clock_gettime(clockid_t kind, timespec* result) noexcept;
extern "C" ssize_t __real_write(int descriptor, const void* bytes, std::size_t count);
extern "C" int __wrap_clock_gettime(clockid_t kind, timespec* result) noexcept {
    if (kind == CLOCK_THREAD_CPUTIME_ID) ++cpu_clock_calls;
    if (kind != CLOCK_MONOTONIC) return __real_clock_gettime(kind, result);
    ++clock_calls;
    errno = EINTR;
    if (clock_mode == ClockMode::kMissing) return -1;
    if (clock_mode == ClockMode::kOverflow) {
        result->tv_sec = static_cast<time_t>(UINT64_MAX / 1000000000ULL + 1);
        result->tv_nsec = 0;
    } else {
        result->tv_sec = 0;
        result->tv_nsec = static_cast<long>(ticks.fetch_add(100));
    }
    return 0;
}
extern "C" ssize_t __wrap_write(int descriptor, const void* bytes, std::size_t count) {
    const std::string_view text{static_cast<const char*>(bytes), count};
    if (descriptor != STDERR_FILENO || !text.starts_with("Glimmer execution copy audit:"))
        return __real_write(descriptor, bytes, count);
    const auto kept = should_truncate && count ? count - 1 : count;
    if (kept > output.size() - output_size) {
        errno = EOVERFLOW;
        return -1;
    }
    for (const auto character : text.substr(0, kept)) output[output_size++] = character;
    errno = ENOSPC;
    return static_cast<ssize_t>(kept);
}
// NOLINTEND(bugprone-reserved-identifier,readability-identifier-naming)

namespace {
using glimmer::interceptor::execution::configure_copy_audit;
using glimmer::interceptor::execution::CopyAudit;
using glimmer::interceptor::execution::CopyRpcAudit;
void disabled_and_unbound() {
    configure_copy_audit(false);
    reset();
    {
        CopyAudit copy(CopyAudit::Api::kDriverUpload, 42);
        CopyAudit::acquired();
        CopyRpcAudit rpc(22, 7);
        expect(!rpc.submission_timestamp(), "disabled audit exposes no transport clock slot");
        rpc.submitted(9);
        rpc.replied(0);
        copy.finish(0);
    }
    expect(clock_calls == 0 && cpu_clock_calls == 0 && captured().empty(),
           "disabled copies read no wall/CPU clock and emit nothing");
    configure_copy_audit(true);
    {
        CopyRpcAudit unbound(25, 7);
        expect(!unbound.submission_timestamp(), "unbound audit exposes no transport clock slot");
        unbound.submitted(9);
        unbound.replied(0);
    }
    expect(clock_calls == 0 && captured().empty(), "unbound RPC cannot invent an API owner");
}
void exact_status_and_errno() {
    reset();
    errno = EDOM;
    {
        CopyAudit copy(CopyAudit::Api::kDriverAsyncUpload, 1024);
        CopyAudit::acquired();
        {
            CopyRpcAudit rpc(22, 41);
            rpc.submitted(8);
            rpc.replied(701);
        }
        copy.finish(701);
    }
    expect(
        errno == EDOM && clock_calls == 6 && cpu_clock_calls == 2 && captured().ends_with("\n") &&
            captured().find("api=2, bytes=1024, entry_ns=100, locked_ns=200, finish_ns=600") !=
                std::string_view::npos &&
            captured().find("has_result=1, status=701, calls=1, dropped=0") !=
                std::string_view::npos &&
            captured().find("operation=22, session=41, request=8, begin_ns=300, submitted_ns=400, "
                            "done_ns=500, has_reply=1, status=701") != std::string_view::npos,
        "exact original failure, request correlation, clocks and errno survive reporting");
}
void transport_timestamp_precedes_binding() {
    for (const auto timestamp : {std::uint64_t{0}, std::uint64_t{350}}) {
        reset();
        {
            CopyAudit copy(CopyAudit::Api::kRuntimeAsync, 1024);
            CopyAudit::acquired();
            {
                CopyRpcAudit rpc(25, 4);
                auto* output = rpc.submission_timestamp();
                expect(output, "active caller owns its trusted transport timestamp slot");
                *output =
                    timestamp;  // Exact synchronous transport output, before sequence binding.
                const auto before = clock_calls.load();
                rpc.submitted(741);
                expect(clock_calls == before, "binding cannot resample after eager delivery");
                rpc.replied(0);
            }
            copy.finish(0);
        }
        const auto expected = "submitted_ns=" + std::to_string(timestamp) + ",";
        expect(captured().find(expected) != std::string_view::npos,
               "transport sample, including invalid zero, is retained rather than fabricated");
    }
}
void overflow_and_unknown_result() {
    reset();
    {
        CopyAudit copy(CopyAudit::Api::kRuntime, 1);
        CopyAudit::acquired();
        for (unsigned int index = 0; index < 20; ++index) {
            CopyRpcAudit rpc(25, 3);
            rpc.submitted(index + 1);
            rpc.replied(0);
        }
        copy.finish(0);
    }
    expect(clock_calls == 27 && output_size <= 4096 &&
               captured().find("calls=8, dropped=12") != std::string_view::npos &&
               captured().find("call=9,") == std::string_view::npos,
           "record overflow stays bounded, explicit and clock-free for dropped calls");
    reset();
    {
        CopyAudit copy(CopyAudit::Api::kRuntimeAsync, 2);
        CopyRpcAudit rpc(22, 8);
    }
    expect(captured().find("has_result=0, status=0") != std::string_view::npos &&
               captured().find("request=0,") != std::string_view::npos &&
               captured().find("has_reply=0, status=0") != std::string_view::npos,
           "missing original result/submission/reply cannot masquerade as success");
}
void clock_and_write_failures() {
    for (const auto mode : {ClockMode::kMissing, ClockMode::kOverflow}) {
        reset();
        clock_mode = mode;
        errno = EDOM;
        {
            CopyAudit copy(CopyAudit::Api::kRuntime, 8);
            CopyAudit::acquired();
            copy.finish(42);
        }
        expect(clock_calls == 1 && captured().empty() && errno == EDOM,
               "unavailable/overflowed clock cannot fabricate timing or alter errno");
    }
    reset();
    should_truncate = true;
    errno = EDOM;
    {
        CopyAudit copy(CopyAudit::Api::kRuntime, 9);
        CopyAudit::acquired();
        copy.finish(701);
    }
    expect(errno == EDOM && !captured().empty() && !captured().ends_with("\n"),
           "partial diagnostic write stays partial evidence, never changes caller result/errno");
}
void nested_tls_and_fork() {
    reset();
    {
        CopyAudit outer(CopyAudit::Api::kDriverUpload, 12);
        CopyAudit::acquired();
        const auto before = clock_calls.load();
        const auto before_cpu = cpu_clock_calls.load();
        std::thread other([] {
            CopyRpcAudit rpc(22, 98);
            rpc.submitted(99);
        });
        other.join();
        {
            CopyAudit nested(CopyAudit::Api::kRuntimeAsync, 1);
            nested.finish(0);
        }
        expect(clock_calls == before && cpu_clock_calls == before_cpu,
               "foreign TLS/nested scope cannot borrow or replace the API timer");
        const auto child = ::fork();
        expect(child >= 0, "fork a CPU-only diagnostic negative control");
        if (!child) {
            {
                CopyAudit copy(CopyAudit::Api::kRuntime, 2);
                CopyAudit::acquired();
                CopyRpcAudit rpc(25, 3);
                rpc.submitted(4);
                copy.finish(0);
                outer.finish(0);
            }
            ::_exit(clock_calls == before && cpu_clock_calls == before_cpu ? 0 : 1);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork child rejects inherited timers without clocks or output");
        outer.finish(0);
    }
    expect(captured().find("api=1, bytes=12") != std::string_view::npos &&
               captured().find("api=4,") == std::string_view::npos,
           "only the original owning caller emits its bounded observation");
}
void rpc_lifecycle() {
    using Metrics = glimmer::interceptor::execution::RpcMetrics;
    reset();
    std::mutex mutex;
    {
        std::unique_lock lock(mutex);
        Metrics::Scope disabled(nullptr, lock, {}, true);
        expect(!disabled.observes_delivery(), "disabled scope requests no delivery observation");
        disabled.sending();
        disabled.submitted(1, true);
        expect(disabled.phase_clock() == 0, "disabled wait phase has no clock");
        disabled.phase(Metrics::Phase::kWaitCall, {});
        disabled.received();
    }
    expect(clock_calls == 0 && cpu_clock_calls == 0, "disabled RPC metrics read no clocks");
    char name[] = "/tmp/glimmer-rpc-metrics-XXXXXX";
    const auto* directory = ::mkdtemp(name);
    expect(directory != nullptr, "fresh owned RPC output directory");
    {
        Metrics metrics(directory, 4);
        errno = EDOM;
        const auto begin = metrics.entry();
        {
            std::unique_lock lock(mutex);
            Metrics::Scope boundary(&metrics, lock, begin, true);
            {
                Metrics::Scope rpc(&metrics, lock, {}, false, {7, 11});
                rpc.sending();
                expect(rpc.observes_delivery(), "enabled RPC observes delivery without clocks");
                const auto before_submit = clock_calls.load();
                rpc.submitted(9, true);
                expect(clock_calls == before_submit + 1, "delivery bit adds no submission clock");
                const auto before = clock_calls.load();
                expect(rpc.phase_clock() == 0, "non-frontier RPC does not collect wait clocks");
                rpc.phase(Metrics::Phase::kWaitCall, {});
                expect(clock_calls == before, "non-frontier wait phases are inert");
                rpc.received();
            }
            {
                Metrics::Scope rpc(&metrics, lock, {}, false, {8, 11});
                rpc.sending();
                const auto before_submit = clock_calls.load();
                rpc.submitted(10, false);
                expect(clock_calls == before_submit + 1, "unresolved delivery adds no clock");
                rpc.received();
            }
        }
        expect(metrics.valid() && metrics.size() == 3 && metrics.record(0).wire == 9 &&
                   metrics.record(0).boundary == metrics.record(2).serial && errno == EDOM,
               "RPC associates exact boundary/session/wire and preserves errno");
        metrics.seal();
        expect(metrics.publish() && !metrics.publish(), "atomic fresh RPC output cannot overwrite");
        const auto path =
            std::string(directory) + "/frontend-rpc-" + std::to_string(::getpid()) + ".json";
        std::ifstream published(path);
        const std::string text{std::istreambuf_iterator<char>{published}, {}};
        published.close();
        expect(
            text.find("\"submit_delivery\":{\"schema\":1,\"fields\":[\"serial\",\"session\","
                      "\"wire\",\"fully_flushed\"],\"records\":[[2,7,9,true],[3,8,10,false]]}") !=
                std::string::npos,
            "delivery addon binds exact serial/session/wire and preserves unresolved false");
        for (std::size_t index = 0; index < metrics.size(); ++index) {
            const auto& row = metrics.record(index);
            const auto suffix = "," + std::to_string(row.cpu_enter) + "," +
                                std::to_string(row.cpu_finish) + "," +
                                std::to_string(index == 2 ? 0 : 3) + "]";
            expect(text.find(suffix) != std::string::npos,
                   "internal delivery bits cannot escape into original lifecycle flags");
        }
        expect(::unlink(path.c_str()) == 0, "remove owned temporary diagnostic");
        const auto phases =
            std::string(directory) + "/frontend-phases-" + std::to_string(::getpid()) + ".json";
        expect(::unlink(phases.c_str()) == 0, "both independently published ledgers are required");
    }
    {
        Metrics metrics(directory, 1);
        std::unique_lock lock(mutex);
        { Metrics::Scope first(&metrics, lock, metrics.entry(), true); }
        { Metrics::Scope overflow(&metrics, lock, metrics.entry(), true); }
        expect(!metrics.valid() && metrics.size() == 1, "RPC capacity overflow fails closed");
    }
    for (unsigned int scenario = 0; scenario < 6; ++scenario) {
        Metrics metrics(directory, 1);
        std::unique_lock lock(mutex);
        {
            Metrics::Scope rpc(&metrics, lock, {}, false, {7, 137});
            rpc.sending();
            rpc.submitted(9);
            lock.unlock();
            const auto begin = rpc.phase_clock();
            const auto end = rpc.phase_clock();
            if (scenario != 1) lock.lock();
            rpc.phase(Metrics::Phase::kWaitCall, {begin, end});
            if (scenario == 1) lock.lock();
            if (scenario == 2) rpc.phase(Metrics::Phase::kPollTake, {begin, end});
            if (scenario == 3) rpc.phase(Metrics::Phase::kPollTake, {end + 1, end});
            if (scenario == 4) rpc.phase(static_cast<Metrics::Phase>(5), {end, end});
            if (scenario == 5) {
                for (unsigned int i = 0; i < 4; ++i) {
                    const auto left = rpc.phase_clock();
                    rpc.phase(Metrics::Phase::kPollTake, {left, rpc.phase_clock()});
                }
            }
            rpc.received();
        }
        expect(metrics.valid() == (scenario == 0),
               "completed phases require lock, order, kind and bounded capacity");
        if (scenario == 0) {
            expect(metrics.phase_size() == 1 &&
                       metrics.phase_record(0).serial == metrics.record(0).serial,
                   "unlocked local timestamps join the exact completed RPC");
        }
    }
    {
        Metrics metrics(directory, 1);
        metrics.seal();
        const auto base =
            std::string(directory) + "/frontend-rpc-" + std::to_string(::getpid()) + ".json";
        const auto phases =
            std::string(directory) + "/frontend-phases-" + std::to_string(::getpid()) + ".json";
        {
            std::ofstream marker(base);
            marker << "existing";
        }
        expect(!metrics.publish() && std::filesystem::is_regular_file(phases) &&
                   std::filesystem::file_size(base) == 8,
               "phase success/base failure cannot overwrite or complete an existing marker");
        expect(::unlink(base.c_str()) == 0 && ::unlink(phases.c_str()) == 0,
               "remove both owned partial-publication controls");
    }
    {
        Metrics metrics(directory, 2);
        const auto child = ::fork();
        expect(child >= 0, "fork RPC metrics negative control");
        if (!child) {
            const auto before = clock_calls.load();
            std::unique_lock lock(mutex);
            { Metrics::Scope inherited(&metrics, lock, metrics.entry(), true); }
            ::_exit(clock_calls == before && metrics.size() == 0 ? 0 : 1);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork cannot record inherited RPC state");
        std::unique_lock lock(mutex);
        clock_mode = ClockMode::kMissing;
        { Metrics::Scope failed(&metrics, lock, metrics.entry(), true); }
        expect(!metrics.valid(), "missing RPC clock cannot fabricate a record");
    }
    expect(::rmdir(directory) == 0, "remove owned temporary directory");
    reset();
}
void host_range_lifecycle() {
    using Metrics = glimmer::interceptor::execution::RpcMetrics;
    using Caller = Metrics::HostRangeCaller;
    reset();
    std::mutex mutex;
    std::unique_lock lock(mutex);
    unsigned int calls{};
    const auto callback = [&](Metrics::HostTrace* trace) noexcept {
        expect(trace == nullptr, "coarse/disabled callback has no detailed observer");
        ++calls;
        errno = ERANGE;
        return false;
    };
    expect(!Metrics::Scope::host_range(nullptr, lock, Caller::kBlasValue, callback) && calls == 1 &&
               errno == ERANGE && clock_calls == 0 && cpu_clock_calls == 0,
           "disabled helper invokes the original once without clocks or changed outcome");
    char name[] = "/tmp/glimmer-host-range-XXXXXX";
    const auto* directory = ::mkdtemp(name);
    expect(directory != nullptr, "fresh helper diagnostic directory");
    {
        Metrics metrics(directory, 4);
        {
            Metrics::Scope boundary(&metrics, lock, metrics.entry(), true);
            for (const auto caller : {Caller::kBlasValue, Caller::kBlasStream}) {
                for (const bool result : {true, false}) {
                    const auto before = clock_calls.load();
                    const auto before_cpu = cpu_clock_calls.load();
                    expect(Metrics::Scope::host_range(&metrics, lock, caller,
                                                      [&](Metrics::HostTrace* trace) noexcept {
                                                          expect(!trace, "coarse-only observer");
                                                          errno = EDOM;
                                                          return result;
                                                      }) == result &&
                               errno == EDOM && clock_calls == before + 2 &&
                               cpu_clock_calls == before_cpu,
                           "exact helper bool and errno survive two local wall-clock samples");
                }
            }
        }
        expect(metrics.valid() && metrics.phase_size() == 4 && metrics.size() == 1,
               "both callers retain success and failure spans without RPC records");
        for (std::size_t index = 0; index < 4; ++index) {
            const auto& span = metrics.phase_record(index);
            const auto& boundary = metrics.record(0);
            expect(span.serial == boundary.serial && span.phase == index + 5 &&
                       boundary.locked <= span.begin && span.begin < span.end &&
                       span.end <= boundary.finish,
                   "helper span is contained in its exact API boundary");
        }
        const auto child = ::fork();
        expect(child >= 0, "fork helper observation negative control");
        if (!child) {
            const auto before = clock_calls.load();
            const auto result =
                Metrics::Scope::host_range(&metrics, lock, Caller::kBlasValue, callback);
            ::_exit(!result && clock_calls == before && metrics.phase_size() == 4 ? 0 : 1);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork helper cannot inherit observer ownership");
        metrics.seal();
        const auto before = clock_calls.load();
        expect(!Metrics::Scope::host_range(&metrics, lock, Caller::kBlasValue, callback) &&
                   clock_calls == before && metrics.phase_size() == 4,
               "sealed observer leaves helper semantics unchanged without clocks");
    }
    for (unsigned int scenario = 0; scenario < 7; ++scenario) {
        Metrics metrics(directory, 1);
        {
            Metrics::Scope boundary(&metrics, lock, metrics.entry(), true);
            if (scenario == 0) lock.unlock();
            const auto caller = scenario == 1 ? static_cast<Caller>(99) : Caller::kBlasValue;
            if (scenario == 2) clock_mode = ClockMode::kMissing;
            if (scenario == 3) ticks = 1;
            if (scenario == 4) {
                std::thread other([&] {
                    std::mutex foreign;
                    std::unique_lock other_lock(foreign);
                    expect(!Metrics::Scope::host_range(&metrics, other_lock, caller, callback),
                           "foreign thread keeps original helper result");
                });
                other.join();
            } else if (scenario == 6) {
                std::mutex foreign;
                std::unique_lock other_lock(foreign);
                expect(!Metrics::Scope::host_range(&metrics, other_lock, caller, callback),
                       "different exclusion keeps original helper result");
            } else {
                for (unsigned int index = 0; index < (scenario == 5 ? 5U : 1U); ++index)
                    expect(!Metrics::Scope::host_range(&metrics, lock, caller, callback),
                           "observer failures never change callback result");
            }
            if (scenario == 0) lock.lock();
            clock_mode = ClockMode::kNormal;
        }
        expect(!metrics.valid() && metrics.phase_size() <= 4,
               "helper lock/caller/clock/order/thread/capacity/exclusion failures invalidate "
               "evidence");
    }
    {
        Metrics metrics(directory, 1);
        const auto before = clock_calls.load();
        expect(!Metrics::Scope::host_range(&metrics, lock, Caller::kBlasValue, callback) &&
                   !metrics.valid() && metrics.phase_size() == 0 && clock_calls == before,
               "unbound helper cannot invent an API parent");
    }
    expect(::rmdir(directory) == 0, "remove owned helper diagnostic directory");
    reset();
}
void host_range_detail() {
    using Metrics = glimmer::interceptor::execution::RpcMetrics;
    using Stage = Metrics::HostStage;
    reset();
    char name[] = "/tmp/glimmer-host-detail-XXXXXX";
    const auto* directory = ::mkdtemp(name);
    expect(directory, "fresh detailed observer directory");
    std::mutex mutex;
    std::unique_lock lock(mutex);
    {
        Metrics metrics(directory, 4, true);
        {
            Metrics::Scope boundary(&metrics, lock, metrics.entry(), true);
            expect(Metrics::Scope::host_range(
                       &metrics, lock, Metrics::HostRangeCaller::kBlasValue,
                       [](Metrics::HostTrace* trace) noexcept {
                           expect(trace, "enabled callback receives its stack trace");
                           struct Descriptor {
                               ~Descriptor() {
                                   ticks.fetch_add(700);
                               }
                           } descriptor;
                           struct EnterCleanup {
                               Metrics::HostTrace* trace;
                               ~EnterCleanup() {
                                   trace->transition(Stage::kCloseOther);
                               }
                           } marker{trace};
                           trace->transition(Stage::kParse);
                           for (const auto bytes : {ssize_t{-1}, ssize_t{4096}, ssize_t{12}}) {
                               trace->transition(Stage::kRead);
                               trace->transition(Stage::kParse);
                               trace->read_result(bytes);
                           }
                           trace->hint_result(false);
                           errno = EDOM;
                           return true;
                       }) &&
                       errno == EDOM,
                   "detailed observation preserves callback and cleanup errno");
        }
        expect(metrics.valid() && metrics.host_size() == 1 && metrics.phase_size() == 1,
               "one detail row accompanies the coarse helper");
        const auto& row = metrics.host_record(0);
        const auto& coarse = metrics.phase_record(0);
        std::uint64_t wall{}, cpu{};
        for (const auto value : row.wall) wall += value;
        for (const auto value : row.cpu) cpu += value;
        expect(row.serial == coarse.serial && coarse.begin <= row.begin && row.end <= coarse.end &&
                   wall == row.end - row.begin && cpu == row.cpu_end - row.cpu_begin &&
                   row.reads == 3 && row.bytes == 4108 && row.closing == 1 && row.wall[3] == 800 &&
                   row.hint_attempted == 1 && row.hint_hit == 0,
               "shared stamps partition time and include descriptor cleanup after marker");
        {
            Metrics::Scope boundary(&metrics, lock, metrics.entry(), true);
            expect(!Metrics::Scope::host_range(
                       &metrics, lock, Metrics::HostRangeCaller::kBlasStream,
                       [](Metrics::HostTrace* trace) noexcept { return trace == nullptr; }),
                   "early rejection retains its false outcome");
        }
        expect(metrics.valid() && metrics.host_size() == 2 && metrics.host_record(1).closing == 0 &&
                   metrics.host_record(1).reads == 0,
               "early rejection does not invent a descriptor close or read");
    }
    for (unsigned int scenario = 0; scenario < 4; ++scenario) {
        Metrics metrics(directory, 1, true);
        {
            Metrics::Scope boundary(&metrics, lock, metrics.entry(), true);
            for (unsigned int index = 0; index < (scenario == 2 ? 2U : 1U); ++index) {
                expect(
                    Metrics::Scope::host_range(&metrics, lock, Metrics::HostRangeCaller::kBlasValue,
                                               [&](Metrics::HostTrace* trace) noexcept {
                                                   if (scenario == 0)
                                                       clock_mode = ClockMode::kMissing;
                                                   if (scenario == 1)
                                                       trace->transition(static_cast<Stage>(99));
                                                   if (scenario == 3) {
                                                       trace->hint_result(false);
                                                       trace->hint_result(true);
                                                   }
                                                   errno = ERANGE;
                                                   return true;
                                               }) &&
                        errno == ERANGE,
                    "detail failure does not change helper result");
                clock_mode = ClockMode::kNormal;
            }
        }
        expect(!metrics.valid() && metrics.host_size() <= 1,
               "detail clock/stage/capacity failures invalidate only evidence");
    }
    expect(::rmdir(directory) == 0, "remove detailed observer directory");
    reset();
}
}  // namespace

int main() {
    try {
        disabled_and_unbound();
        exact_status_and_errno();
        transport_timestamp_precedes_binding();
        overflow_and_unknown_result();
        clock_and_write_failures();
        nested_tls_and_fork();
        rpc_lifecycle();
        host_range_lifecycle();
        host_range_detail();
        configure_copy_audit(false);
        std::cout << "Execution copy audit checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
