#include "dnn_dropout_fixture.h"

#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_dropout;
using Kind = CudaDnnObjectKind;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "dropout native operation preserves original success");
}
std::shared_ptr<CudaDnnDropoutMutation> reserve(CudaDnnRegistry& registry, std::uint64_t id) {
    auto result = registry.begin_dropout_mutation(id);
    expect(result.result.disposition == CudaDnnDisposition::kComplete &&
               result.result.status == CUDA_SUCCESS && result.mutation,
           "exclusive mutation reservation is acquired without invoking the SDK");
    return std::move(result.mutation);
}
void bound(CudaDnnRegistry& registry, std::uint64_t id, const CudaDnnDropoutBinding& expected) {
    const auto view = registry.borrow_dropout(id);
    expect(view && view->object.kind == Kind::kDropout &&
               view->binding.context.lock() == expected.context.lock() &&
               view->binding.allocation == expected.allocation &&
               view->binding.offset == expected.offset && view->binding.bytes == expected.bytes,
           "borrow exposes exact CPU state provenance without owning its native allocation");
}
void lifecycle() {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto created = registry.create(Kind::kDropout);
    success(created);
    expect(fixture::state.creates == 1 && !fixture::state.sets && !fixture::state.restores &&
               !fixture::driver::control.context_creates.load(),
           "original dropout creation has no hidden setter or CUDA context");
    bound(registry, created.id, {});
    {
        auto view = registry.borrow_dropout(created.id);
        expect(view &&
                   registry.begin_dropout_mutation(created.id).result.disposition ==
                       CudaDnnDisposition::kDeferred &&
                   registry.destroy(created.id).disposition == CudaDnnDisposition::kDeferred,
               "immutable consumers exclude both mutation and descriptor destruction");
    }
    {
        auto mutation = reserve(registry, created.id);
        expect(mutation->native() && !mutation->old_binding().allocation &&
                   !registry.borrow_dropout(created.id) &&
                   !registry.borrow(created.id, Kind::kDropout) &&
                   registry.begin_dropout_mutation(created.id).result.disposition ==
                       CudaDnnDisposition::kDeferred &&
                   registry.destroy(created.id).disposition == CudaDnnDisposition::kDeferred,
               "unissued reservation excludes all immutable and mutable paths");
    }
    bound(registry, created.id, {});
    auto context = std::make_shared<const int>(9);
    const CudaDnnDropoutBinding first{context, 17, 8, 32}, second{context, 18, 16, 48};
    {
        auto mutation = reserve(registry, created.id);
        expect(mutation->claim() && mutation->mark_issued(), "reservation issues once");
        mutation->finish(CUDNN_STATUS_SUCCESS, first, CUDA_SUCCESS);
        expect(mutation->completion_result() &&
                   mutation->completion_result()->original_status == CUDNN_STATUS_SUCCESS &&
                   mutation->completion_result()->status == CUDA_SUCCESS,
               "completed mutation preserves original and physical status separately");
        expect(!registry.borrow_dropout(created.id) &&
                   registry.destroy(created.id).disposition == CudaDnnDisposition::kDeferred,
               "successful host completion does not release an outstanding physical token");
    }
    bound(registry, created.id, first);
    {
        auto cancelled = reserve(registry, created.id);
        expect(cancelled->old_binding().allocation == first.allocation,
               "unissued rebinding can inspect existing provenance");
    }
    bound(registry, created.id, first);
    {
        auto mutation = reserve(registry, created.id);
        expect(mutation->claim() && !mutation->claim() && mutation->mark_issued() &&
                   !mutation->mark_issued(),
               "one mutation cannot be admitted or issued twice");
        mutation->finish(CUDNN_STATUS_SUCCESS, second, CUDA_SUCCESS);
    }
    bound(registry, created.id, second);
    context.reset();
    {
        const auto stale = registry.borrow_dropout(created.id);
        expect(stale && stale->binding.context.expired() &&
                   stale->binding.allocation == second.allocation,
               "provenance does not extend the context lifetime or silently clear stale identity");
    }
    success(registry.destroy(created.id));
    expect(fixture::state.destroys == 1 && !fixture::state.sets && !fixture::state.restores &&
               registry.destroy(created.id).status == CUDA_ERROR_INVALID_HANDLE,
           "registry token tests execute no fake native mutation and destroy once");
    success(registry.shutdown());
}
void boundaries() {
    fixture::reset();
    for (bool create_missing : {false, true}) {
        auto sdk = fixture::dispatch();
        if (create_missing)
            sdk.dropout_create = nullptr;
        else
            sdk.dropout_destroy = nullptr;
        CudaDnnRegistry registry(sdk);
        const auto result = registry.create(Kind::kDropout);
        expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
                   !fixture::state.creates,
               "creation requires original create and destruction callbacks before native call");
        success(registry.shutdown());
    }
    CudaDnnRegistry registry(fixture::dispatch(), 2);
    fixture::state.create_status = CUDNN_STATUS_ALLOC_FAILED;
    const auto failed = registry.create(Kind::kDropout);
    expect(failed.original_status == CUDNN_STATUS_ALLOC_FAILED && !failed.id &&
               !registry.snapshot().objects,
           "known empty create failure retains original status without fabricating ownership");
    fixture::state.create_status = CUDNN_STATUS_SUCCESS;
    const auto id = registry.create(Kind::kDropout).id;
    const auto tensor = registry.create(Kind::kTensor).id;
    expect(registry.create(Kind::kDropout).status == CUDA_ERROR_OUT_OF_MEMORY &&
               registry.begin_dropout_mutation(tensor).result.status == CUDA_ERROR_INVALID_HANDLE &&
               registry.begin_dropout_mutation(999).result.status == CUDA_ERROR_INVALID_HANDLE &&
               !registry.borrow_dropout(tensor) && !registry.borrow_dropout(999),
           "bounded creation and wrong-kind/stale mutation reject before native access");
    const auto foreign = std::async(std::launch::async, [&] {
                             return registry.begin_dropout_mutation(id).result;
                         }).get();
    expect(foreign.status == CUDA_ERROR_UNKNOWN, "foreign owner thread cannot reserve mutation");
    const auto child = ::fork();
    expect(child >= 0, "fork isolated inherited-owner check");
    if (!child)
        ::_exit(!registry.borrow_dropout(id) &&
                        registry.begin_dropout_mutation(id).result.status == CUDA_ERROR_UNKNOWN
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork child cannot reuse inherited metadata authority");
    success(registry.shutdown());
    expect(fixture::state.destroys == 1, "ordinary mixed registry closes exact dropout ownership");
}
void uncertain(unsigned int mode) {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    if (mode < 3) {
        if (mode == 0) {
            fixture::state.partial_create = true;
            fixture::state.create_status = CUDNN_STATUS_ALLOC_FAILED;
        }
        if (mode == 1) fixture::state.null_create = true;
        if (mode == 2) {
            success(registry.create(Kind::kDropout));
            fixture::state.duplicate_create = true;
        }
        const auto result = registry.create(Kind::kDropout);
        expect(!result.id && registry.snapshot().is_quarantined,
               "partial/null/duplicate native ownership cannot publish a descriptor");
    } else {
        const auto id = registry.create(Kind::kDropout).id;
        if (mode == 3) {
            fixture::state.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
            expect(registry.destroy(id).original_status == CUDNN_STATUS_INTERNAL_ERROR,
                   "uncertain native destruction preserves the original error");
        } else {
            auto mutation = reserve(registry, id);
            expect(mutation->claim() && mutation->mark_issued(), "uncertain reservation issues");
            if (mode != 4)
                mutation->finish(mode == 5   ? CUDNN_STATUS_INTERNAL_ERROR
                                 : mode == 7 ? CUDNN_STATUS_BAD_PARAM
                                 : mode == 8 ? CUDNN_STATUS_ALLOC_FAILED
                                 : mode == 9 ? CUDNN_STATUS_NOT_SUPPORTED
                                             : CUDNN_STATUS_SUCCESS,
                                 {}, mode == 6 ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS);
            if (mode == 6)
                expect(
                    mutation->completion_result() &&
                        mutation->completion_result()->original_status == CUDNN_STATUS_SUCCESS &&
                        mutation->completion_result()->status == CUDA_ERROR_INVALID_CONTEXT &&
                        mutation->completion_result()->disposition == CudaDnnDisposition::kRejected,
                    "failed physical proof cannot be projected as successful original status");
            mutation.reset();
            expect(!registry.borrow_dropout(id), "uncertain mutation cannot become a consumer");
        }
    }
    const auto attempts = fixture::state.destroys;
    expect(registry.shutdown().disposition != CudaDnnDisposition::kComplete &&
               registry.shutdown().disposition != CudaDnnDisposition::kComplete &&
               fixture::state.destroys == attempts && !registry.snapshot().is_closed,
           "uncertain metadata cleanup is retained and never retried");
}
}  // namespace
int main() {
    try {
        lifecycle();
        boundaries();
        for (unsigned int mode = 0; mode != 10; ++mode) {
            const auto child = ::fork();
            expect(child >= 0, "fork bounded intentional-quarantine fixture");
            if (!child) {
                try {
                    uncertain(mode);
                    ::_exit(0);
                } catch (const std::exception& error) {
                    std::cerr << "dropout quarantine " << mode << ": " << error.what() << '\n';
                    ::_exit(1);
                }
            }
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "all isolated dropout uncertain-owner assertions pass");
        }
        std::cout << "Dropout registry exclusivity and provenance checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
