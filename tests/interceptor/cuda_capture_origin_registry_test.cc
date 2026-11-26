#include "internal/cuda_capture_origin_registry.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <thread>

namespace {
using namespace glimmer::interceptor;
const auto kContext = reinterpret_cast<CUcontext>(0x1000);
const auto kOtherContext = reinterpret_cast<CUcontext>(0x2000);
const auto kStream = reinterpret_cast<CUstream>(0x3000);
const auto kOtherStream = reinterpret_cast<CUstream>(0x4000);
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void overlap_and_capabilities() {
    CudaCaptureOriginRegistry registry;
    CudaCaptureOriginRegistry foreign;
    CudaCaptureOriginToken first, second, wrong;
    require(
        registry.observe_origin(kContext, kStream, false, first) == CUDA_SUCCESS &&
            registry.observe_origin(kOtherContext, kOtherStream, true, second) == CUDA_SUCCESS &&
            foreign.observe_origin(kContext, kStream, false, wrong) == CUDA_SUCCESS,
        "observe actual contexts and original stream ABIs before native issue");
    unsigned int calls{};
    auto state = CU_STREAM_CAPTURE_STATUS_ACTIVE;
    const CudaCaptureOriginRegistry::Query query = [&](CUcontext actual_context,
                                                       CUstream actual_stream, bool is_ptds,
                                                       CUstreamCaptureStatus* output) {
        ++calls;
        require((actual_context == kContext && actual_stream == kStream && !is_ptds) ||
                    (actual_context == kOtherContext && actual_stream == kOtherStream && is_ptds),
                "query preserves exact native context, stream and ABI");
        *output = actual_stream == kStream ? CU_STREAM_CAPTURE_STATUS_NONE : state;
        return CUDA_SUCCESS;
    };
    require(registry.reconcile(wrong, query) == CUDA_ERROR_INVALID_VALUE && calls == 0,
            "foreign capability runs no query");
    require(registry.verify_none(query) == CUDA_ERROR_NOT_READY && calls == 2 &&
                registry.retained_origins() == 1,
            "NONE on one origin cannot complete the active overlapping capture");
    require(registry.reconcile(first, query) == CUDA_ERROR_INVALID_VALUE && calls == 2,
            "retired origin capability is stale");
    state = CU_STREAM_CAPTURE_STATUS_INVALIDATED;
    require(registry.reconcile(second, query) == CUDA_ERROR_NOT_READY &&
                registry.retained_origins() == 1,
            "invalidated capture is not NONE regardless of original end error");
    state = CU_STREAM_CAPTURE_STATUS_NONE;
    require(registry.verify_none(query) == CUDA_SUCCESS && registry.retained_origins() == 0,
            "only every original NONE query permits inactive proof");
    CudaCaptureOriginToken next;
    require(registry.observe_origin(kOtherContext, kOtherStream, false, next) == CUDA_SUCCESS,
            "stream reuse creates a new owned generation");
    const auto before = calls;
    require(registry.reconcile(second, query) == CUDA_ERROR_INVALID_VALUE && calls == before,
            "old generation cannot query or retire a reused handle");
    require(registry.reconcile(next,
                               [&](CUcontext, CUstream, bool, CUstreamCaptureStatus* output) {
                                   *output = CU_STREAM_CAPTURE_STATUS_NONE;
                                   return CUDA_SUCCESS;
                               }) == CUDA_SUCCESS,
            "fresh capability can retire its exact origin");
}

void destroyed_origin_capabilities() {
    CudaCaptureOriginRegistry registry(2), foreign;
    CudaCaptureOriginToken destroyed, live, wrong, reused;
    require(registry.observe_origin(kContext, kStream, false, destroyed) == CUDA_SUCCESS &&
                registry.observe_origin(kOtherContext, kOtherStream, false, live) == CUDA_SUCCESS &&
                foreign.observe_origin(kContext, kStream, false, wrong) == CUDA_SUCCESS,
            "prepare exact destroy generation and independent live origin");
    require(registry.retire_destroyed({}) == CUDA_ERROR_INVALID_VALUE &&
                registry.retire_destroyed(wrong) == CUDA_ERROR_INVALID_VALUE &&
                registry.retained_origins() == 2,
            "foreign/missing destroy capability cannot erase an actual origin");
    require(registry.retire_destroyed(destroyed) == CUDA_SUCCESS &&
                registry.retire_destroyed(destroyed) == CUDA_ERROR_INVALID_VALUE &&
                registry.retained_origins() == 1,
            "destroy success retires one exact origin without querying it or completing others");
    unsigned int queries{};
    require(registry.verify_none(
                [&](CUcontext context, CUstream stream, bool, CUstreamCaptureStatus* output) {
                    ++queries;
                    require(context == kOtherContext && stream == kOtherStream,
                            "destroyed handle is never queried through old capture metadata");
                    *output = CU_STREAM_CAPTURE_STATUS_ACTIVE;
                    return CUDA_SUCCESS;
                }) == CUDA_ERROR_NOT_READY &&
                queries == 1 && registry.retained_origins() == 1,
            "remaining ACTIVE origin keeps independent capture ownership");
    require(registry.observe_origin(kContext, kStream, false, reused) == CUDA_SUCCESS &&
                registry.retire_destroyed(destroyed) == CUDA_ERROR_INVALID_VALUE &&
                registry.retire_destroyed(reused) == CUDA_SUCCESS &&
                registry.retire_destroyed(live) == CUDA_SUCCESS,
            "recycled handle has a new generation, never an old destruction permission");
    CudaCaptureOriginToken implicit;
    require(registry.observe_origin(kContext, nullptr, false, implicit) == CUDA_SUCCESS &&
                registry.retire_destroyed(implicit) == CUDA_ERROR_INVALID_VALUE &&
                registry.verify_none([&](CUcontext, CUstream, bool, CUstreamCaptureStatus* output) {
                    require(registry.retire_destroyed(implicit) == CUDA_ERROR_INVALID_VALUE,
                            "verification excludes destruction mutation");
                    *output = CU_STREAM_CAPTURE_STATUS_NONE;
                    return CUDA_SUCCESS;
                }) == CUDA_SUCCESS,
            "implicit streams cannot be destroyed by explicit stream capability");
}

void default_stream_abi_and_capacity() {
    struct Case {
        CUstream input;
        bool is_ptds;
        CUstream alias;
        bool alias_ptds;
    };
    const std::array<Case, 5> cases{{
        {nullptr, false, CU_STREAM_LEGACY, true},
        {CU_STREAM_LEGACY, true, nullptr, false},
        {nullptr, true, CU_STREAM_PER_THREAD, false},
        {CU_STREAM_PER_THREAD, false, nullptr, true},
        {kStream, true, kStream, false},
    }};
    for (const auto& item : cases) {
        CudaCaptureOriginRegistry registry(1);
        CudaCaptureOriginToken first, alias, rejected;
        require(
            registry.observe_origin(kContext, item.input, item.is_ptds, first) == CUDA_SUCCESS &&
                registry.observe_origin(kContext, item.alias, item.alias_ptds, alias) ==
                    CUDA_SUCCESS &&
                registry.retained_origins() == 1,
            "equivalent default or explicit streams share one existing generation");
        require(registry.observe_origin(kOtherContext, kOtherStream, false, rejected) ==
                        CUDA_ERROR_OUT_OF_MEMORY &&
                    !rejected,
                "bounded registration rejects before untracked native issue");
        require(
            registry.observe_origin(kContext, kStream, false, first) == CUDA_ERROR_INVALID_VALUE,
            "nonempty output capability cannot be replaced");
        require(registry.reconcile(alias,
                                   [&](CUcontext actual, CUstream actual_stream, bool is_ptds,
                                       CUstreamCaptureStatus* output) {
                                       require(
                                           actual == kContext && actual_stream == item.input &&
                                               is_ptds == item.is_ptds,
                                           "canonical identity does not change original query ABI");
                                       *output = CU_STREAM_CAPTURE_STATUS_NONE;
                                       return CUDA_SUCCESS;
                                   }) == CUDA_SUCCESS,
                "same-caller default origin can be queried through its original ABI");
        require(registry.reconcile(first, {}) == CUDA_ERROR_INVALID_VALUE,
                "same-generation alias becomes stale after NONE");
    }
    CudaCaptureOriginRegistry registry(2);
    CudaCaptureOriginToken legacy, ptds, invalid;
    require(registry.observe_origin(nullptr, kStream, false, invalid) == CUDA_ERROR_INVALID_VALUE &&
                registry.observe_origin(kContext, nullptr, false, legacy) == CUDA_SUCCESS &&
                registry.observe_origin(kContext, nullptr, true, ptds) == CUDA_SUCCESS &&
                registry.retained_origins() == 2,
            "null legacy and null PTDS are distinct actual origins");
}

void query_failure_and_reentrancy() {
    for (unsigned int mode = 0; mode < 5; ++mode) {
        CudaCaptureOriginRegistry registry;
        CudaCaptureOriginToken token;
        require(registry.observe_origin(kContext, kStream, false, token) == CUDA_SUCCESS,
                "prepare failure origin");
        unsigned int calls{};
        const CudaCaptureOriginRegistry::Query query =
            [&](CUcontext, CUstream, bool, CUstreamCaptureStatus* output) -> CUresult {
            ++calls;
            if (mode == 0) {
                *output = CU_STREAM_CAPTURE_STATUS_NONE;
                return CUDA_ERROR_INVALID_CONTEXT;
            }
            if (mode == 4) {
                *output = CU_STREAM_CAPTURE_STATUS_NONE;
                return CUDA_ERROR_NOT_READY;
            }
            if (mode == 1) {
                *output = static_cast<CUstreamCaptureStatus>(3);
            } else if (mode == 2) {
                throw std::runtime_error("uncertain original capture query");
            }
            return CUDA_SUCCESS;
        };
        const auto expected = mode == 0   ? CUDA_ERROR_INVALID_CONTEXT
                              : mode == 4 ? CUDA_ERROR_NOT_READY
                                          : CUDA_ERROR_UNKNOWN;
        require(registry.verify_none(query) == expected && calls == 1 &&
                    registry.retained_origins() == 1 &&
                    registry.reconcile(token, query) == expected &&
                    registry.verify_none(query) == expected && calls == 1,
                "failed, unknown, throwing or unwritten query never authorizes thaw or replay");
        CudaCaptureOriginToken rejected;
        require(registry.observe_origin(kContext, kOtherStream, false, rejected) == expected &&
                    !rejected,
                "failed registry rejects new native origin ownership");
        CUcontext context{};
        require(registry.lookup_context(kStream, false, nullptr, context) == expected && !context,
                "failed registry cannot recover a context or mutate its output");
    }
    CudaCaptureOriginRegistry registry;
    CudaCaptureOriginToken token;
    require(registry.observe_origin(kContext, kStream, false, token) == CUDA_SUCCESS,
            "prepare reentrancy origin");
    unsigned int nested_calls{};
    const CudaCaptureOriginRegistry::Query nested = [&](CUcontext, CUstream, bool,
                                                        CUstreamCaptureStatus*) {
        ++nested_calls;
        return CUDA_SUCCESS;
    };
    require(registry.verify_none([&](CUcontext, CUstream, bool, CUstreamCaptureStatus* output) {
        CudaCaptureOriginToken rejected;
        CUcontext context{};
        require(registry.observe_origin(kContext, kOtherStream, false, rejected) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    registry.reconcile(token, nested) == CUDA_ERROR_INVALID_VALUE &&
                    registry.retire_destroyed(token) == CUDA_ERROR_INVALID_VALUE &&
                    registry.verify_none(nested) == CUDA_ERROR_INVALID_VALUE &&
                    registry.lookup_context(kStream, false, nullptr, context) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    !context && nested_calls == 0,
                "query callbacks cannot invalidate iteration through nested mutation");
        *output = CU_STREAM_CAPTURE_STATUS_NONE;
        return CUDA_SUCCESS;
    }) == CUDA_SUCCESS &&
                registry.retained_origins() == 0,
            "reentrant rejection preserves the outer original query result");
}

void caller_lifetimes() {
    CudaCaptureOriginRegistry registry;
    CudaCaptureOriginToken owner, other;
    std::latch observed(1), may_verify(1);
    CUresult observed_result{}, owner_result{};
    unsigned int owner_calls{};
    std::thread thread([&] {
        observed_result = registry.observe_origin(kContext, nullptr, true, owner);
        observed.count_down();
        may_verify.wait();
        owner_result = registry.reconcile(
            owner, [&](CUcontext, CUstream input, bool is_ptds, CUstreamCaptureStatus* output) {
                ++owner_calls;
                if (input != nullptr || !is_ptds) {
                    return CUDA_ERROR_INVALID_VALUE;
                }
                *output = CU_STREAM_CAPTURE_STATUS_NONE;
                return CUDA_SUCCESS;
            });
    });
    observed.wait();
    unsigned int foreign_calls{};
    const CudaCaptureOriginRegistry::Query foreign = [&](CUcontext, CUstream, bool,
                                                         CUstreamCaptureStatus* output) {
        ++foreign_calls;
        *output = CU_STREAM_CAPTURE_STATUS_NONE;
        return CUDA_SUCCESS;
    };
    const auto foreign_result = registry.verify_none(foreign);
    const auto other_result = registry.observe_origin(kContext, CU_STREAM_PER_THREAD, false, other);
    const auto retained = registry.retained_origins();
    may_verify.count_down();
    thread.join();
    require(observed_result == CUDA_SUCCESS && foreign_result == CUDA_ERROR_NOT_READY &&
                foreign_calls == 0 && other_result == CUDA_SUCCESS && retained == 2 &&
                owner_result == CUDA_SUCCESS && owner_calls == 1,
            "foreign PTDS cannot be queried and callers retain distinct default origins");
    require(registry.verify_none(foreign) == CUDA_SUCCESS && foreign_calls == 1,
            "owner-proven NONE allows another caller to finish remaining origins");

    CudaCaptureOriginRegistry expired;
    CudaCaptureOriginToken lost;
    CUresult lost_result{};
    std::thread exiting(
        [&] { lost_result = expired.observe_origin(kContext, CU_STREAM_PER_THREAD, false, lost); });
    exiting.join();
    require(lost_result == CUDA_SUCCESS &&
                expired.verify_none(foreign) == CUDA_ERROR_INVALID_HANDLE && foreign_calls == 1 &&
                expired.retained_origins() == 1,
            "dead caller identity cannot be replaced by a new host thread or foreign null stream");

    CudaCaptureOriginRegistry explicit_stream;
    CudaCaptureOriginToken explicit_origin;
    std::thread normal([&] {
        lost_result = explicit_stream.observe_origin(kContext, kStream, true, explicit_origin);
    });
    normal.join();
    require(lost_result == CUDA_SUCCESS && explicit_stream.verify_none(foreign) == CUDA_SUCCESS &&
                foreign_calls == 2,
            "explicit native streams remain queryable after initiating thread exit");
}

void recorded_context_lookup() {
    CudaCaptureOriginRegistry registry;
    CudaCaptureOriginToken explicit_origin, legacy, ptds, overlapping_default;
    require(registry.observe_origin(kContext, kStream, false, explicit_origin) == CUDA_SUCCESS &&
                registry.observe_origin(kContext, nullptr, false, legacy) == CUDA_SUCCESS &&
                registry.observe_origin(kContext, nullptr, true, ptds) == CUDA_SUCCESS &&
                registry.observe_origin(kOtherContext, nullptr, true, overlapping_default) ==
                    CUDA_SUCCESS,
            "retain real typed origin contexts before lookup");
    for (const auto stream : {kStream, CU_STREAM_LEGACY, CU_STREAM_PER_THREAD}) {
        CUcontext context{};
        require(registry.lookup_context(stream, true, kContext, context) == CUDA_SUCCESS &&
                    context == kContext && registry.retained_origins() == 4,
                "canonical alias lookup preserves known context without retirement");
        require(
            registry.lookup_context(stream, true, kContext, context) == CUDA_ERROR_INVALID_VALUE &&
                context == kContext,
            "nonempty context output cannot be replaced");
    }
    CUcontext context{};
    require(registry.lookup_context(nullptr, true, kOtherContext, context) == CUDA_SUCCESS &&
                context == kOtherContext,
            "implicit PTDS lookup matches actual current context, not another origin");
    context = nullptr;
    require(registry.lookup_context(nullptr, false, kOtherContext, context) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                !context &&
                registry.lookup_context(nullptr, true, nullptr, context) ==
                    CUDA_ERROR_INVALID_CONTEXT &&
                !context &&
                registry.lookup_context(kOtherStream, false, nullptr, context) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                !context,
            "missing or mismatched default origins cannot supply guessed context");
    CudaCaptureOriginToken ambiguous;
    require(registry.observe_origin(kOtherContext, kStream, true, ambiguous) == CUDA_SUCCESS &&
                registry.lookup_context(kStream, false, nullptr, context) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                !context,
            "ambiguous explicit stream identity rejects without output mutation");
    require(registry.verify_none([](CUcontext, CUstream, bool, CUstreamCaptureStatus* output) {
        *output = CU_STREAM_CAPTURE_STATUS_NONE;
        return CUDA_SUCCESS;
    }) == CUDA_SUCCESS &&
                registry.lookup_context(kStream, false, nullptr, context) ==
                    CUDA_ERROR_INVALID_HANDLE &&
                !context,
            "retired origins cannot be reused as capture-end identity");

    CudaCaptureOriginRegistry foreign;
    CudaCaptureOriginToken foreign_token;
    std::latch observed(1), may_exit(1);
    CUresult observed_status{};
    std::thread thread([&] {
        observed_status = foreign.observe_origin(kContext, nullptr, true, foreign_token);
        observed.count_down();
        may_exit.wait();
    });
    observed.wait();
    const auto foreign_status = foreign.lookup_context(nullptr, true, kContext, context);
    may_exit.count_down();
    thread.join();
    require(
        observed_status == CUDA_SUCCESS && foreign_status == CUDA_ERROR_NOT_READY && !context &&
            foreign.lookup_context(nullptr, true, kContext, context) == CUDA_ERROR_INVALID_HANDLE &&
            !context,
        "foreign or dead PTDS callers cannot authorize original end");
}
}  // namespace
int main() {
    try {
        overlap_and_capabilities();
        destroyed_origin_capabilities();
        default_stream_abi_and_capacity();
        query_failure_and_reentrancy();
        caller_lifetimes();
        recorded_context_lookup();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
