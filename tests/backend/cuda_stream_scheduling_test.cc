#include "cuda_execution_domain_fixture.h"
#include "internal/cuda_profile_cache.h"

#include <bit>
#include <optional>

namespace {
namespace fixture = glimmer::test::cuda_domain;
using namespace glimmer::backend::cuda;
using fixture::expect;
using fixture::success;
void verify_sdk(const CudaSessionResult& result, std::string_view phase) {
    try {
        success(result);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string(phase) + ": " + error.what());
    }
}
struct CreatedStream {
    CUstream stream{};
    int requested{}, actual{};
};
std::vector<CreatedStream> created;
CUresult range_error{}, create_error{}, query_error{};
bool has_partial_create{}, has_wrong_priority{}, has_invalid_range{};
unsigned int range_calls{}, create_calls{}, query_calls{};
void reset() {
    fixture::driver = {};
    created.clear();
    range_error = create_error = query_error = CUDA_SUCCESS;
    has_partial_create = has_wrong_priority = has_invalid_range = false;
    range_calls = create_calls = query_calls = 0;
}
// Exact original SDK callback signatures.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CUresult CUDAAPI priority_range(int* least, int* greatest) {
    ++range_calls;
    expect(fixture::driver.current, "query range only in the selected context");
    *least = 0;
    *greatest = has_invalid_range ? 1 : -3;
    return std::exchange(range_error, CUDA_SUCCESS);
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CUresult CUDAAPI create_priority(CUstream* output, unsigned int flags, int priority) {
    ++create_calls;
    const auto status = std::exchange(create_error, CUDA_SUCCESS);
    if (status != CUDA_SUCCESS && !has_partial_create) return status;
    expect(fixture::stream_create(output, flags) == CUDA_SUCCESS, "bounded native stream factory");
    created.push_back({*output, priority, std::clamp(priority, -3, 0)});
    return status;
}
CUresult CUDAAPI get_priority(CUstream stream, int* output) {
    ++query_calls;
    const auto found = std::ranges::find(created, stream, &CreatedStream::stream);
    expect(found != created.end(), "query only an exact created stream");
    *output = found->actual + (has_wrong_priority ? 1 : 0);
    return std::exchange(query_error, CUDA_SUCCESS);
}
CudaSessionDispatch dispatch() {
    auto result = fixture::session_dispatch();
    result.context.priority_range = priority_range;
    result.stream_create_priority = create_priority;
    result.stream_get_priority = get_priority;
    return result;
}
CudaSessionStreamConfig streams(std::uint32_t priority) {
    return {.uses_default_proxy = true, .scheduling_priority = priority};
}
void healthy_priorities_and_projection() {
    for (const auto configured : {0U, 1U, 3U, 100U, UINT32_MAX}) {
        reset();
        CudaExecutionSession session({1, 17}, 3, dispatch(), fixture::limits(1000), {}, {}, {},
                                     false, streams(configured));
        verify_sdk(session.initialize(0, {CU_CTX_MAP_HOST}), "healthy initialization");
        const auto native = -static_cast<int>(std::min(configured, 3U));
        expect(created.size() == 1 && created[0].requested == native && range_calls == 1 &&
                   query_calls == 1,
               "default proxy uses actual bounded native priority, including configured zero");
        std::vector<std::uint64_t> ids;
        for (const auto requested :
             {std::optional<int>{}, std::optional<int>{0}, std::optional<int>{1},
              std::optional<int>{-1}, std::optional<int>{99}, std::optional<int>{-99},
              std::optional<int>{INT32_MIN}, std::optional<int>{INT32_MAX}}) {
            const auto result = session.create_stream(CU_STREAM_DEFAULT, requested);
            verify_sdk(result, "healthy source stream creation");
            expect(result.id &&
                       result.address ==
                           std::bit_cast<std::uint32_t>(std::clamp(requested.value_or(0), -3, 0)) &&
                       created.back().requested == native && created.back().actual == native &&
                       created.back().stream != created.front().stream,
                   "hidden policy cannot replace original source priority/clamping metadata");
            ids.push_back(result.id);
        }
        for (const auto id : ids) verify_sdk(session.destroy_stream(id), "healthy stream destroy");
        session.close();
        verify_sdk(session.shutdown(), "healthy shutdown");
        expect(fixture::driver.context_creates == 1 && fixture::driver.context_destroys == 1 &&
                   !fixture::driver.live_resources() && create_calls == 9 && query_calls == 9,
               "every original/private stream remains owned through normal context cleanup");
    }
}
void missing_dispatch_rejects_before_sdk() {
    for (unsigned int mode = 0; mode < 6; ++mode) {
        reset();
        auto sdk = dispatch();
        auto configuration = streams(0);
        CudaSessionPersistentConfig persistent;
        std::shared_ptr<CudaProfileCache> profile;
        if (mode == 0) configuration.uses_default_proxy = false;
        if (mode == 1) sdk.context.priority_range = nullptr;
        if (mode == 2) sdk.stream_create_priority = nullptr;
        if (mode == 3) sdk.stream_get_priority = nullptr;
        if (mode == 4) persistent.workers = 1;
        if (mode == 5) profile = std::make_shared<CudaProfileCache>();
        bool rejected{};
        try {
            CudaExecutionSession session({1, 17}, 3, sdk, fixture::limits(1000), persistent, {},
                                         profile, false, configuration);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && !fixture::driver.context_creates && !range_calls && !create_calls,
               "invalid cold priority configuration cannot silently use unprioritized streams");
    }
}
void failed_sdk_ownership() {
    for (const auto during_initialization : {true, false}) {
        for (unsigned int mode = 0; mode < 6; ++mode) {
            const bool retains_private_proxy = during_initialization && mode >= 2 && mode <= 4;
            if (retains_private_proxy) {
                const auto child = ::fork();
                expect(child >= 0, "isolate deliberate unknown native proxy ownership");
                if (child) {
                    int status{};
                    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                               WEXITSTATUS(status) == 0,
                           "isolated retention oracle passes, not healthy cleanup evidence");
                    continue;
                }
            }
            reset();
            CudaExecutionSession session({1, 17}, 3, dispatch(), fixture::limits(1000), {}, {}, {},
                                         false, streams(UINT32_MAX));
            if (!during_initialization) success(session.initialize(0, {CU_CTX_MAP_HOST}));
            if (mode == 0) range_error = CUDA_ERROR_INVALID_DEVICE;
            if (mode == 1 || mode == 2) create_error = CUDA_ERROR_OUT_OF_MEMORY;
            if (mode == 2) has_partial_create = true;
            if (mode == 3) query_error = CUDA_ERROR_INVALID_HANDLE;
            if (mode == 4) has_wrong_priority = true;
            if (mode == 5) has_invalid_range = true;
            const auto result = during_initialization
                                    ? session.initialize(0, {CU_CTX_MAP_HOST})
                                    : session.create_stream(CU_STREAM_NON_BLOCKING, -99);
            const auto expected = mode == 0                ? CUDA_ERROR_INVALID_DEVICE
                                  : mode == 1 || mode == 2 ? CUDA_ERROR_OUT_OF_MEMORY
                                  : mode == 3              ? CUDA_ERROR_INVALID_HANDLE
                                                           : CUDA_ERROR_UNKNOWN;
            expect(result.status == expected && !result.id && session.snapshot().has_context &&
                       fixture::driver.context_creates == 1 &&
                       create_calls ==
                           (during_initialization ? 0U : 1U) + (mode == 0 || mode == 5 ? 0U : 1U),
                   "policy/native failure reports its exact error without publishing a new owner");
            if (during_initialization || mode == 3 || mode == 4)
                expect(
                    result.original_status == CUDA_SUCCESS,
                    "extra policy/postcondition failure retains actual original creation success");
            else if (mode == 1 || mode == 2)
                expect(result.original_status == CUDA_ERROR_OUT_OF_MEMORY,
                       "actual source stream creation failure preserves native provenance");
            if (!during_initialization && (mode == 0 || mode == 1)) {
                const auto retry = session.create_stream(CU_STREAM_DEFAULT);
                success(retry);
                success(session.destroy_stream(retry.id));
            } else {
                const auto attempts = create_calls;
                expect(
                    (during_initialization ? session.initialize(0, {CU_CTX_MAP_HOST})
                                           : session.create_stream(CU_STREAM_DEFAULT))
                                .status != CUDA_SUCCESS &&
                        create_calls == attempts,
                    "partial initialization/uncertain source creation never retries native work");
            }
            session.close();
            if (retains_private_proxy) {
                expect(session.shutdown().status == CUDA_ERROR_UNKNOWN &&
                           session.snapshot().has_context && session.snapshot().is_quarantined &&
                           !fixture::driver.context_destroys && fixture::driver.live_resources(),
                       "uncertain private proxy retains its native owner through shutdown");
                const auto cleanup_calls = fixture::driver.cleanup_calls();
                expect(session.shutdown().status == CUDA_ERROR_UNKNOWN &&
                           fixture::driver.cleanup_calls() == cleanup_calls,
                       "a retry cannot destroy an uncertain private proxy");
                // Explicit isolated uncertainty. Do not run destructors or
                // call this case leak-free/normal native-resource retirement.
                ::_exit(EXIT_SUCCESS);
            }
            verify_sdk(session.shutdown(),
                       "failure shutdown initialization=" + std::to_string(during_initialization) +
                           " mode=" + std::to_string(mode));
            expect(
                fixture::driver.context_destroys == 1 && !fixture::driver.live_resources(),
                "known isolated context owns partial streams; this is not shared recovery proof");
        }
    }
}
}  // namespace
int main() {
    try {
        healthy_priorities_and_projection();
        missing_dispatch_rejects_before_sdk();
        failed_sdk_ownership();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
