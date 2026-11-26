#include "dnn_rnn_attention_fixture.h"

#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_rnn_attention;
using Kind = CudaDnnObjectKind;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "original metadata success");
}
CudaDnnRnnState rnn(std::uint64_t dropout = 0) {
    return {.algorithm = 7,
            .cell_mode = 3,
            .bias_mode = 2,
            .direction = 1,
            .input_mode = 1,
            .type = 31,
            .math_precision = 30,
            .math_type = 3,
            .input_size = INT32_MIN,
            .hidden_size = INT32_MAX,
            .projection_size = -7,
            .layers = 0,
            .dropout = dropout,
            .flags = UINT32_MAX};
}
CudaDnnAttnState attn(std::uint64_t first = 0, std::uint64_t second = 0) {
    return {.mode = UINT32_MAX,
            .heads = INT32_MIN,
            .scale = -0.0,
            .type = 31,
            .precision = 30,
            .math_type = 3,
            .dropouts = {first, second},
            .query_size = INT32_MIN,
            .key_size = INT32_MAX,
            .value_size = -1,
            .query_projection = 0,
            .key_projection = 1,
            .value_projection = 2,
            .output_projection = 3,
            .max_query_sequence = 4,
            .max_key_sequence = 5,
            .max_batch = 6,
            .max_beam = 7};
}
void blocked_dropout(CudaDnnRegistry& registry, std::uint64_t id) {
    const auto before = fixture::dropout::state.destroys;
    const auto result = registry.destroy(id);
    expect(result.disposition == CudaDnnDisposition::kRejected &&
               result.status == CUDA_ERROR_INVALID_VALUE && !result.original_status &&
               !registry.snapshot().is_quarantined && fixture::dropout::state.destroys == before,
           "referenced dropout rejects immediately and cleanly, never Deferred");
}
void ordinary() {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto parent = registry.create(Kind::kRnn).id, attention = registry.create(Kind::kAttn).id;
    const auto first = registry.create(Kind::kDropout).id,
               second = registry.create(Kind::kDropout).id,
               third = registry.create(Kind::kDropout).id;
    expect(parent && attention && first && second && third && !fixture::state.rnn_sets &&
               !fixture::state.attn_sets,
           "native defaults are created without hidden setter or getter");
    success(registry.set_rnn(parent, rnn(first)));
    success(registry.set_attn(attention, attn(first, first)));
    expect(
        fixture::state.last_rnn.enums == std::array<std::uint32_t, 8>{7, 3, 2, 1, 1, 31, 30, 3} &&
            fixture::state.last_rnn.sizes ==
                std::array<std::int32_t, 4>{INT32_MIN, INT32_MAX, -7, 0} &&
            fixture::state.last_rnn.flags == UINT32_MAX &&
            fixture::state.last_attn.mode == UINT32_MAX &&
            fixture::state.last_attn.heads == INT32_MIN &&
            fixture::state.last_attn.scale_bits == 0x8000000000000000ULL &&
            fixture::state.last_attn.enums == std::array<std::uint32_t, 3>{31, 30, 3} &&
            fixture::state.last_attn.sizes ==
                std::array<int, 11>{INT32_MIN, INT32_MAX, -1, 0, 1, 2, 3, 4, 5, 6, 7} &&
            fixture::state.last_attn.dropouts[0] == fixture::state.last_attn.dropouts[1] &&
            !registry.snapshot().borrowed,
        "all ABI fields and equal positional pointers survive without durable immutable borrows");
    blocked_dropout(registry, first);
    {
        auto mutation = registry.begin_dropout_mutation(first);
        expect(mutation.mutation && mutation.result.status == CUDA_SUCCESS,
               "logical dependency does not prevent subsequent dropout mutation");
        const auto calls = fixture::control.callbacks.load();
        expect(registry.set_rnn(parent, rnn(first)).disposition == CudaDnnDisposition::kDeferred &&
                   registry.set_attn(attention, attn(first, second)).disposition ==
                       CudaDnnDisposition::kDeferred &&
                   fixture::control.callbacks.load() == calls,
               "pending physical mutation defers a supplied dependency before SDK issue");
        expect(mutation.mutation->claim() && mutation.mutation->mark_issued(),
               "issue direct token once");
        mutation.mutation->finish(CUDNN_STATUS_SUCCESS, {}, CUDA_SUCCESS);
    }
    {
        const auto held = registry.borrow(parent, Kind::kRnn);
        expect(
            held &&
                registry.set_rnn(parent, rnn(second)).disposition ==
                    CudaDnnDisposition::kDeferred &&
                registry.destroy(parent).disposition == CudaDnnDisposition::kDeferred &&
                registry.shutdown().disposition == CudaDnnDisposition::kDeferred,
            "external parent lease excludes mutation and shutdown without releasing dependencies");
    }
    // shutdown marks closing even while deferred; use a separate fixture for
    // successful rebind rather than inventing a reopening operation.
    success(registry.shutdown());
    expect(fixture::state.destruction.size() == 5 && fixture::state.destroys == 2 &&
               fixture::dropout::state.destroys == 3,
           "dependency-safe shutdown overrides reverse creation order and destroys exactly once");
}
void rebind() {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto parent = registry.create(Kind::kRnn).id, attention = registry.create(Kind::kAttn).id;
    const auto first = registry.create(Kind::kDropout).id,
               second = registry.create(Kind::kDropout).id,
               third = registry.create(Kind::kDropout).id;
    success(registry.set_rnn(parent, rnn(first)));
    success(registry.set_attn(attention, attn(first, first)));
    success(registry.set_rnn(parent, rnn(second)));
    blocked_dropout(registry, first);
    success(registry.set_attn(attention, attn(second, third)));
    success(registry.destroy(first));
    success(registry.set_attn(attention, attn(second, second)));
    success(registry.destroy(third));
    blocked_dropout(registry, second);
    success(registry.set_rnn(parent, rnn()));
    success(registry.set_attn(attention, attn()));
    success(registry.destroy(second));
    const auto calls = fixture::control.callbacks.load();
    auto invalid = rnn();
    invalid.algorithm = 8;
    expect(registry.set_rnn(parent, invalid).status == CUDA_ERROR_INVALID_VALUE &&
               registry.set_rnn(parent, rnn(first)).status == CUDA_ERROR_INVALID_HANDLE &&
               registry.set_rnn(parent, rnn(attention)).status == CUDA_ERROR_INVALID_HANDLE &&
               registry.set_attn(parent, attn()).status == CUDA_ERROR_INVALID_HANDLE &&
               calls == fixture::control.callbacks.load() && !registry.snapshot().is_quarantined,
           "raw representation, stale and wrong-kind errors reject cleanly before native issue");
    const auto child = ::fork();
    expect(child >= 0, "fork inherited registry");
    if (!child) ::_exit(registry.set_rnn(parent, rnn()).status == CUDA_ERROR_UNKNOWN ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "inherited registry authority cannot issue native configuration");
    expect(std::async(std::launch::async, [&] { return registry.set_rnn(parent, rnn()); })
                   .get()
                   .status == CUDA_ERROR_UNKNOWN,
           "foreign thread cannot issue native metadata");
    success(registry.shutdown());
    expect(fixture::foundation::is_resource_free(),
           "metadata requires no Driver or compute operations");
}
void reentry() {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto dependency = registry.create(Kind::kDropout).id;
    unsigned int entries{};
    fixture::state.callback = [&] {
        ++entries;
        expect(
            registry.create(Kind::kRnn).status == CUDA_ERROR_NOT_READY &&
                registry.destroy(dependency).status == CUDA_ERROR_NOT_READY &&
                registry.set_rnn(999, {}).status == CUDA_ERROR_NOT_READY &&
                registry.set_attn(999, {}).status == CUDA_ERROR_NOT_READY &&
                registry.begin_dropout_mutation(dependency).result.status == CUDA_ERROR_NOT_READY &&
                !registry.borrow(dependency, Kind::kDropout) &&
                registry.shutdown().disposition == CudaDnnDisposition::kDeferred,
            "new lifecycle/setter native callbacks cannot reenter any metadata ownership path");
    };
    const auto parent = registry.create(Kind::kRnn).id, attention = registry.create(Kind::kAttn).id;
    success(registry.set_rnn(parent, rnn(dependency)));
    success(registry.set_attn(attention, attn(dependency, dependency)));
    success(registry.destroy(parent));
    success(registry.shutdown());
    fixture::state.callback = {};
    expect(entries == 6,
           "guard spans both creates, both setters, explicit destroy and shutdown destroy");
    fixture::reset();
    CudaDnnRegistry closing(fixture::dispatch());
    const auto id = closing.create(Kind::kRnn).id;
    fixture::state.callback = [&] {
        closing.close();
        expect(!fixture::state.destroys && closing.snapshot().is_closing,
               "close during callback records intent without invalidating accepted ownership");
    };
    success(closing.set_rnn(id, rnn()));
    fixture::state.callback = {};
    expect(closing.set_rnn(id, rnn()).disposition == CudaDnnDisposition::kRejected,
           "later mutation cannot reopen a closing owner");
    success(closing.shutdown());
}
void missing() {
    for (unsigned int absent = 0; absent != 6; ++absent) {
        fixture::reset();
        auto sdk = fixture::dispatch();
        if (absent == 0) sdk.rnn_create = nullptr;
        if (absent == 1) sdk.rnn_destroy = nullptr;
        if (absent == 2) sdk.rnn_set_v8 = nullptr;
        if (absent == 3) sdk.attn_create = nullptr;
        if (absent == 4) sdk.attn_destroy = nullptr;
        if (absent == 5) sdk.attn_set = nullptr;
        CudaDnnRegistry registry(sdk);
        auto result = registry.create(absent < 3 ? Kind::kRnn : Kind::kAttn);
        if (absent == 2) result = registry.set_rnn(result.id, rnn());
        if (absent == 5) result = registry.set_attn(result.id, attn());
        expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
                   !registry.snapshot().is_quarantined,
               "missing callback is clean policy rejection");
        success(registry.shutdown());
    }
}
void poison(unsigned int variant) {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto kind = variant % 2 ? Kind::kAttn : Kind::kRnn;
    if (variant >= 4) {
        if (variant < 6) {
            fixture::state.partial_create = true;
            fixture::state.create_status = CUDNN_STATUS_ALLOC_FAILED;
        } else if (variant < 8)
            fixture::state.null_create = true;
        else {
            success(registry.create(kind));
            fixture::state.duplicate_create = true;
        }
        const auto result = registry.create(kind);
        expect(result.disposition == CudaDnnDisposition::kRejected &&
                   registry.snapshot().is_quarantined,
               "partial/null/duplicate original construction retains uncertain ownership");
    } else {
        const auto parent = registry.create(kind).id;
        std::array<std::uint64_t, 4> dependencies{};
        for (auto& id : dependencies) id = registry.create(Kind::kDropout).id;
        success(kind == Kind::kRnn
                    ? registry.set_rnn(parent, rnn(dependencies[0]))
                    : registry.set_attn(parent, attn(dependencies[0], dependencies[1])));
        if (variant < 2) {
            fixture::control.set_status.store(CUDNN_STATUS_BAD_PARAM);
            const auto result =
                kind == Kind::kRnn
                    ? registry.set_rnn(parent, rnn(dependencies[2]))
                    : registry.set_attn(parent, attn(dependencies[2], dependencies[3]));
            expect(result.original_status == CUDNN_STATUS_BAD_PARAM,
                   "partially applied old/new rebind preserves original native error");
        } else {
            fixture::state.destroy_status = CUDNN_STATUS_ALLOC_FAILED;
            expect(registry.destroy(parent).original_status == CUDNN_STATUS_ALLOC_FAILED,
                   "failed dependent destroy preserves native status and dependency owners");
        }
        expect(registry.snapshot().is_quarantined, "issued error quarantines dependency owner");
        for (const auto id : dependencies)
            expect(registry.destroy(id).disposition == CudaDnnDisposition::kRejected,
                   "old and candidate dependencies are not released after ambiguous mutation");
    }
    const auto destroyed = fixture::state.destroys;
    expect(registry.shutdown().disposition != CudaDnnDisposition::kComplete &&
               fixture::state.destroys == destroyed && !fixture::dropout::state.destroys,
           "uncertain dependency graph neither retries native cleanup nor destroys dropouts");
    ::_exit(0);
}
}  // namespace
int main() {
    try {
        ordinary();
        rebind();
        reentry();
        missing();
        for (unsigned int variant = 0; variant != 10; ++variant) {
            const auto child = ::fork();
            expect(child >= 0, "fork independent dependency poison case");
            if (!child) poison(variant);
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "isolated dependency poison case passes");
        }
        std::cout << "RNN/attention dependency registry checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
