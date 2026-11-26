#include "internal/cuda_nccl_primary_domain.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;

void expect(bool condition, std::string_view message) {
    if (condition) return;
    std::cerr << "NCCL primary domain: " << message << '\n';
    std::_Exit(1);
}
enum class Call { kGet, kDevice, kRetain, kSet, kCreate, kQuery, kDestroy, kRelease };
struct Context {
    bool alive{true};
};
struct Stream {
    bool alive{}, ready{true};
};
struct Native {
    Context source, primary, other;
    CUcontext current{reinterpret_cast<CUcontext>(&source)};
    std::array<Stream, 16> streams{};
    std::vector<Call> calls;
    std::function<void()> callback;
    std::size_t fail_at{}, corrupt_at{}, next_stream{}, retains{}, releases{}, destroys{};
    bool partial_output{}, null_primary{}, null_stream{}, duplicate_stream{};
    CUresult visit(Call call) {
        calls.push_back(call);
        if (callback) callback();
        if (corrupt_at == calls.size()) current = reinterpret_cast<CUcontext>(&other);
        return fail_at == calls.size() ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
    }
    CUcontext primary_context() {
        return reinterpret_cast<CUcontext>(&primary);
    }
    CUcontext source_context() {
        return reinterpret_cast<CUcontext>(&source);
    }
    void clear() {
        calls.clear();
        fail_at = 0;
        corrupt_at = 0;
    }
}* native;

CUresult CUDAAPI get_current(CUcontext* output) {
    const auto status = native->visit(Call::kGet);
    *output = native->current;
    return status;
}
CUresult CUDAAPI set_current(CUcontext context) {
    const auto status = native->visit(Call::kSet);
    if (status == CUDA_SUCCESS && native->corrupt_at != native->calls.size())
        native->current = context;
    return status;
}
CUresult CUDAAPI device_get(CUdevice* output, int ordinal) {
    const auto status = native->visit(Call::kDevice);
    expect(ordinal == 3, "requested device ordinal, not current source device");
    *output = 41;
    return status;
}
CUresult CUDAAPI retain(CUcontext* output, CUdevice device) {
    const auto status = native->visit(Call::kRetain);
    expect(device == 41, "retain original resolved device identity");
    if ((status == CUDA_SUCCESS || native->partial_output) && !native->null_primary) {
        ++native->retains;
        *output = native->primary_context();
    }
    return status;
}
CUresult CUDAAPI release(CUdevice device) {
    const auto status = native->visit(Call::kRelease);
    expect(device == 41 && native->current != native->primary_context() && native->retains == 1 &&
               native->releases == 0,
           "release exact retained device once and never while current");
    ++native->releases;
    if (status == CUDA_SUCCESS) native->primary.alive = false;
    return status;
}
CUresult CUDAAPI create_stream(CUstream* output, unsigned flags) {
    const auto status = native->visit(Call::kCreate);
    expect(flags == CU_STREAM_NON_BLOCKING, "every execution stream is explicitly nonblocking");
    if (native->corrupt_at != native->calls.size())
        expect(native->current == native->primary_context(), "stream creation under owned primary");
    if ((status == CUDA_SUCCESS || native->partial_output) && !native->null_stream) {
        if (native->duplicate_stream) {
            *output = reinterpret_cast<CUstream>(&native->streams[0]);
        } else {
            expect(native->next_stream < native->streams.size(), "bounded fake stream inventory");
            auto& stream = native->streams[native->next_stream++];
            stream.alive = true;
            *output = reinterpret_cast<CUstream>(&stream);
        }
    }
    return status;
}
CUresult CUDAAPI query_stream(CUstream stream) {
    const auto status = native->visit(Call::kQuery);
    if (native->corrupt_at != native->calls.size())
        expect(native->current == native->primary_context(), "query under exact primary context");
    const auto& value = *reinterpret_cast<Stream*>(stream);
    expect(value.alive, "query only actual live streams");
    return status != CUDA_SUCCESS ? status : (value.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY);
}
CUresult CUDAAPI destroy_stream(CUstream stream) {
    const auto status = native->visit(Call::kDestroy);
    if (native->corrupt_at != native->calls.size())
        expect(native->current == native->primary_context(), "destroy under exact primary context");
    auto& value = *reinterpret_cast<Stream*>(stream);
    expect(value.alive && value.ready, "stream destruction follows actual successful stop query");
    ++native->destroys;
    if (status == CUDA_SUCCESS) value.alive = false;
    return status;
}
CudaNcclPrimaryDomainDispatch dispatch() {
    return {device_get,  retain,        release,      get_current,
            set_current, create_stream, query_stream, destroy_stream};
}
struct Owner {
    Native state;
    std::unique_ptr<CudaNcclPrimaryDomain> domain;
    explicit Owner(std::size_t maximum_streams = 4) {
        native = &state;
        domain = std::make_unique<CudaNcclPrimaryDomain>(
            dispatch(), CudaNcclPrimaryDomainConfig{701, 3, maximum_streams});
    }
    void initialize() {
        expect(domain->initialize() == CUDA_SUCCESS, "retain actual primary");
    }
    CudaNcclPrimaryStreamCapture stream() {
        auto result = domain->create_stream();
        expect(result.status == CUDA_SUCCESS && result.stream_id && result.domain.stream &&
                   result.domain.stream_lifetime && result.domain.context.lifetime &&
                   result.domain.device_ordinal == 3 && result.domain.context.id == 701 &&
                   result.domain.context.native == state.primary_context(),
               "immutable stream capsule retains exact context and stream");
        return result;
    }
    void shutdown() {
        expect(domain->close() == CUDA_SUCCESS && domain->shutdown() == CUDA_SUCCESS,
               "explicit known cleanup");
    }
};

const std::vector<Call> kInitialize{Call::kGet, Call::kGet,    Call::kDevice, Call::kGet,
                                    Call::kGet, Call::kRetain, Call::kGet};
const std::vector<Call> kCreate{Call::kGet,    Call::kGet, Call::kSet, Call::kGet, Call::kGet,
                                Call::kCreate, Call::kGet, Call::kGet, Call::kSet, Call::kGet};
const std::vector<Call> kRetire{Call::kGet,   Call::kGet, Call::kSet, Call::kGet,     Call::kGet,
                                Call::kQuery, Call::kGet, Call::kGet, Call::kDestroy, Call::kGet,
                                Call::kGet,   Call::kSet, Call::kGet};
const std::vector<Call> kShutdown{Call::kGet, Call::kGet, Call::kRelease, Call::kGet};

void successful_lifecycle() {
    for (const bool no_current : {false, true}) {
        Owner owner;
        if (no_current) owner.state.current = nullptr;
        const auto saved = owner.state.current;
        expect(owner.state.calls.empty(), "constructor has no SDK side effects");
        owner.initialize();
        expect(owner.state.calls == kInitialize && owner.state.current == saved,
               "retain neither changes nor fabricates the service current context");
        const auto count = owner.state.calls.size();
        expect(owner.domain->initialize() == CUDA_ERROR_INVALID_VALUE &&
                   owner.state.calls.size() == count,
               "duplicate initialize does not retain twice");
        auto context = owner.domain->capture_context();
        expect(context.status == CUDA_SUCCESS &&
                   context.context.native == owner.state.primary_context() &&
                   context.context.id == 701 && context.context.lifetime &&
                   owner.state.calls.size() == count,
               "context-only capture is CPU-only");
        owner.state.clear();
        auto first = owner.stream();
        expect(owner.state.calls == kCreate && owner.state.current == saved,
               "stream create has proven entry/switch/restore ledger");
        auto second = owner.stream();
        expect(first.stream_id != second.stream_id && first.domain.stream != second.domain.stream,
               "distinct owned stream identities and native handles");
        const auto first_id = first.stream_id;
        auto stream_only = first.domain.stream_lifetime;
        first = {};
        owner.state.clear();
        expect(owner.domain->release_stream(first_id) == CUDA_ERROR_NOT_READY &&
                   owner.state.calls.empty(),
               "stream-only capsule prevents native retirement");
        std::thread release_copy([hold = std::move(stream_only)]() mutable { hold.reset(); });
        release_copy.join();
        expect(owner.state.calls.empty(), "worker capsule destruction makes no SDK calls");
        expect(owner.domain->release_stream(first_id) == CUDA_SUCCESS &&
                   owner.state.calls == kRetire && owner.state.current == saved,
               "context-only communicator hold permits unrelated exact stream cleanup");
        second = {};
        expect(owner.domain->close() == CUDA_SUCCESS, "close admits no new owners");
        owner.state.clear();
        expect(owner.domain->capture_context().status == CUDA_ERROR_DEINITIALIZED &&
                   owner.domain->create_stream().status == CUDA_ERROR_DEINITIALIZED &&
                   owner.domain->shutdown() == CUDA_ERROR_NOT_READY && owner.state.calls.empty(),
               "native owner capsule, not completed stream, controls context release");
        context = {};
        expect(owner.domain->shutdown() == CUDA_SUCCESS && owner.state.current == saved &&
                   owner.state.destroys == 2 && owner.state.releases == 1,
               "internal stream context holds do not permanently block complete shutdown");
        const auto after = owner.state.calls.size();
        expect(owner.domain->shutdown() == CUDA_SUCCESS, "successful shutdown is idempotent");
        owner.domain.reset();
        expect(owner.state.calls.size() == after, "destructor cannot repeat native cleanup");
    }
}

void quota_and_current_primary() {
    Owner owner(1);
    expect(owner.domain->capture_context().status == CUDA_ERROR_NOT_INITIALIZED &&
               owner.domain->create_stream().status == CUDA_ERROR_NOT_INITIALIZED,
           "uninitialized capture is not authority");
    owner.initialize();
    auto stream = owner.stream();
    const auto id = stream.stream_id;
    owner.state.clear();
    expect(owner.domain->create_stream().status == CUDA_ERROR_OUT_OF_MEMORY &&
               owner.state.calls.empty(),
           "live stream quota rejects before any SDK callback");
    stream = {};
    owner.state.streams[0].ready = false;
    expect(owner.domain->release_stream(id) == CUDA_ERROR_NOT_READY &&
               owner.state.current == owner.state.source_context() && owner.state.destroys == 0,
           "pending stream query restores source context without destruction");
    owner.state.streams[0].ready = true;
    expect(owner.domain->release_stream(id) == CUDA_SUCCESS,
           "ready retry consumes no duplicate destroy");
    owner.state.clear();
    expect(owner.domain->release_stream(id) == CUDA_ERROR_INVALID_HANDLE &&
               owner.domain->release_stream(0) == CUDA_ERROR_INVALID_HANDLE &&
               owner.state.calls.empty(),
           "stale and zero stream IDs issue no SDK");
    auto replacement = owner.stream();
    expect(replacement.stream_id > id, "refunded quota never recycles logical stream identity");
    replacement = {};
    expect(owner.domain->close() == CUDA_SUCCESS, "close before shutdown");
    owner.state.current = owner.state.primary_context();
    owner.state.clear();
    expect(owner.domain->shutdown() == CUDA_ERROR_NOT_READY &&
               owner.state.calls == std::vector{Call::kGet} && owner.state.releases == 0 &&
               owner.state.destroys == 1,
           "cannot final-release current primary or restore its dead handle");
    owner.state.current = nullptr;
    expect(owner.domain->shutdown() == CUDA_SUCCESS && !owner.state.current,
           "explicit safe entry context permits later cleanup");
}

void owner_and_callback_exclusion() {
    Owner owner;
    std::size_t callbacks{};
    owner.state.callback = [&] {
        ++callbacks;
        const auto before = owner.state.calls.size();
        expect(owner.domain->initialize() == CUDA_ERROR_NOT_READY &&
                   owner.domain->capture_context().status == CUDA_ERROR_NOT_READY &&
                   owner.domain->create_stream().status == CUDA_ERROR_NOT_READY &&
                   owner.domain->release_stream(1) == CUDA_ERROR_NOT_READY &&
                   owner.domain->close() == CUDA_ERROR_NOT_READY &&
                   owner.domain->shutdown() == CUDA_ERROR_NOT_READY &&
                   owner.state.calls.size() == before,
               "all methods exclude callback reentry before native entry or CPU mutation");
    };
    owner.initialize();
    auto stream = owner.stream();
    stream = {};
    const auto before = owner.state.calls.size();
    std::thread foreign([&] {
        expect(owner.domain->initialize() == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.domain->capture_context().status == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.domain->create_stream().status == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.domain->release_stream(1) == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.domain->close() == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.domain->shutdown() == CUDA_ERROR_INVALID_CONTEXT,
               "wrong owner rejects all mutable methods");
    });
    foreign.join();
    expect(owner.state.calls.size() == before, "wrong thread issued no SDK");
    const auto child = ::fork();
    expect(child >= 0, "fork ownership isolation");
    if (!child) {
        expect(owner.domain->capture_context().status == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.domain->close() == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.domain->shutdown() == CUDA_ERROR_INVALID_CONTEXT,
               "PID gate precedes inherited owner/synchronization state");
        owner.domain.reset();
        expect(owner.state.calls.size() == before, "fork destructor performs no SDK cleanup");
        std::_Exit(0);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork negative child succeeded");
    owner.shutdown();
    expect(callbacks == owner.state.calls.size(),
           "every callback including cleanup was reentry checked");
}

void constructor_and_empty_cleanup() {
    Native state;
    native = &state;
    for (unsigned invalid = 0; invalid != 11; ++invalid) {
        auto sdk = dispatch();
        CudaNcclPrimaryDomainConfig config{701, 3, 4};
        if (invalid == 0) config.identity = 0;
        if (invalid == 1) config.device_ordinal = -1;
        if (invalid == 2) config.maximum_streams = 0;
        if (invalid == 3) sdk.device_get = nullptr;
        if (invalid == 4) sdk.primary_retain = nullptr;
        if (invalid == 5) sdk.primary_release = nullptr;
        if (invalid == 6) sdk.context_get_current = nullptr;
        if (invalid == 7) sdk.context_set_current = nullptr;
        if (invalid == 8) sdk.stream_create = nullptr;
        if (invalid == 9) sdk.stream_query = nullptr;
        if (invalid == 10) sdk.stream_destroy = nullptr;
        bool rejected{};
        try {
            CudaNcclPrimaryDomain invalid_domain(sdk, config);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && state.calls.empty(), "invalid configuration rejected before SDK");
    }
    Owner empty;
    expect(empty.domain->shutdown() == CUDA_ERROR_NOT_READY, "close explicitly before shutdown");
    empty.shutdown();
    expect(empty.state.calls.empty(), "unused domain teardown requires no CUDA initialization");
    Owner initialized;
    initialized.initialize();
    initialized.state.clear();
    initialized.shutdown();
    expect(initialized.state.calls == kShutdown && initialized.state.releases == 1,
           "context-only domain has a checked exact-once release ledger");
}

enum class Phase { kInitialize, kCreate, kRetire, kShutdown };
enum class Fault { kSdkError, kContextChange };
void isolated_failure(Phase phase, Fault fault, std::size_t position) {
    const bool corrupt = fault == Fault::kContextChange;
    Owner owner;
    std::uint64_t stream_id{};
    if (phase != Phase::kInitialize) owner.initialize();
    if (phase == Phase::kRetire) {
        auto stream = owner.stream();
        stream_id = stream.stream_id;
    }
    if (phase == Phase::kShutdown)
        expect(owner.domain->close() == CUDA_SUCCESS, "close failure target");
    owner.state.clear();
    if (corrupt)
        owner.state.corrupt_at = position;
    else
        owner.state.fail_at = position;
    owner.state.partial_output = true;
    CUresult result{};
    switch (phase) {
        case Phase::kInitialize:
            result = owner.domain->initialize();
            break;
        case Phase::kCreate: {
            const auto created = owner.domain->create_stream();
            result = created.status;
            expect(!created.stream_id && !created.domain.stream &&
                       !created.domain.stream_lifetime && !created.domain.context.native &&
                       !created.domain.context.lifetime,
                   "failed native/proof preparation publishes no resource capsule");
            break;
        }
        case Phase::kRetire:
            result = owner.domain->release_stream(stream_id);
            break;
        case Phase::kShutdown:
            result = owner.domain->shutdown();
            break;
    }
    expect(result == (corrupt ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_OUT_OF_MEMORY),
           "callback/proof failure remains exact and is never success");
    const auto& ledger = phase == Phase::kInitialize ? kInitialize
                         : phase == Phase::kCreate   ? kCreate
                         : phase == Phase::kRetire   ? kRetire
                                                     : kShutdown;
    if (corrupt) {
        const auto expected_calls = position + (ledger[position - 1] == Call::kGet ? 0 : 1);
        expect(owner.state.calls.size() == expected_calls,
               "proof contradiction stops all further callbacks including restoration");
    } else if ((phase == Phase::kCreate && position == 6) ||
               (phase == Phase::kRetire && (position == 6 || position == 9))) {
        expect(owner.state.current == owner.state.source_context(),
               "intact proof permits source restoration after original resource-call error");
    }
    const auto before = owner.state.calls.size();
    const auto releases = owner.state.releases;
    const auto destroys = owner.state.destroys;
    expect(owner.domain->initialize() == result && owner.domain->create_stream().status == result &&
               owner.domain->capture_context().status == result &&
               owner.domain->release_stream(stream_id) == result &&
               owner.domain->close() == result && owner.domain->shutdown() == result &&
               owner.state.calls.size() == before,
           "uncertain resource/proof failure is sticky and no operation retries native cleanup");
    owner.domain.reset();
    expect(owner.state.calls.size() == before && owner.state.releases == releases &&
               owner.state.destroys == destroys,
           "uncertain destructor retains native resources without SDK work");
}
template <typename Callback>
void isolated(Callback callback) {
    const auto child = ::fork();
    expect(child >= 0, "isolate intentional retained ownership");
    if (!child) {
        callback();
        std::_Exit(0);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated retained owner control");
}
void failure_matrix() {
    const std::array phases{Phase::kInitialize, Phase::kCreate, Phase::kRetire, Phase::kShutdown};
    const std::array<std::size_t, 4> lengths{7, 10, 13, 4};
    for (std::size_t index = 0; index != phases.size(); ++index)
        for (std::size_t position = 1; position <= lengths[index]; ++position)
            isolated([&] { isolated_failure(phases[index], Fault::kSdkError, position); });
    // Genuine side-effect contradictions, including outgoing restoration. The
    // first current observation defines the entry context, so it is not a
    // meaningful corruption target without an independent native status error.
    for (const auto position : {2U, 3U, 4U, 5U, 6U, 7U})
        isolated([&] { isolated_failure(Phase::kInitialize, Fault::kContextChange, position); });
    for (const auto position : {2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U})
        isolated([&] { isolated_failure(Phase::kCreate, Fault::kContextChange, position); });
    for (const auto position : {2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U, 10U, 11U, 12U, 13U})
        isolated([&] { isolated_failure(Phase::kRetire, Fault::kContextChange, position); });
    for (const auto position : {2U, 3U, 4U})
        isolated([&] { isolated_failure(Phase::kShutdown, Fault::kContextChange, position); });
}

void malformed_success_and_destructor() {
    for (unsigned variant = 0; variant != 5; ++variant)
        isolated([&] {
            Owner owner;
            if (variant == 0) {
                owner.state.null_primary = true;
                expect(owner.domain->initialize() == CUDA_ERROR_UNKNOWN,
                       "SUCCESS with NULL primary retains uncertain ownership");
            } else {
                owner.initialize();
                if (variant == 1) {
                    owner.state.null_stream = true;
                    expect(owner.domain->create_stream().status == CUDA_ERROR_UNKNOWN,
                           "SUCCESS with NULL stream retains uncertain owner");
                } else if (variant == 2) {
                    auto first = owner.stream();
                    first = {};
                    owner.state.duplicate_stream = true;
                    expect(owner.domain->create_stream().status == CUDA_ERROR_UNKNOWN,
                           "duplicate native stream is not independently destroyable");
                } else if (variant == 3) {
                    auto held = owner.domain->capture_context();
                    owner.domain.reset();
                    expect(held.context.native == owner.state.primary_context() &&
                               held.context.lifetime && owner.state.releases == 0,
                           "manager destruction cannot invalidate exported communicator owner");
                    held = {};
                } else {
                    std::unique_ptr<CudaNcclPrimaryDomain> expired;
                    std::thread creator([&] {
                        expired = std::make_unique<CudaNcclPrimaryDomain>(
                            dispatch(), CudaNcclPrimaryDomainConfig{702, 3, 1});
                    });
                    creator.join();
                    std::thread replacement([&] {
                        CudaNcclPrimaryDomain fresh(dispatch(), {703, 3, 1});
                        expect(expired->initialize() == CUDA_ERROR_INVALID_CONTEXT &&
                                   expired->capture_context().status == CUDA_ERROR_INVALID_CONTEXT,
                               "new thread incarnation cannot reuse expired owner authority");
                    });
                    replacement.join();
                }
            }
            const auto calls = owner.state.calls.size();
            owner.domain.reset();
            expect(owner.state.calls.size() == calls && owner.state.releases == 0 &&
                       owner.state.destroys == 0,
                   "CPU destruction never resets, releases or destroys uncertain native owners");
        });
}
}  // namespace

int main() {
    constructor_and_empty_cleanup();
    successful_lifecycle();
    quota_and_current_primary();
    owner_and_callback_exclusion();
    failure_matrix();
    malformed_success_and_destructor();
    std::cout << "NCCL primary domain: 6 groups, 69 isolated retained-failure controls passed\n";
}
