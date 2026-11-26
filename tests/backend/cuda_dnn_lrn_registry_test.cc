#include "dnn_fixture.h"

#include <bit>
#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace sdk = glimmer::tests::dnn;
using Kind = CudaDnnObjectKind;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "native LRN success retains both status domains");
}
void original(const CudaDnnResult& result, cudnnStatus_t status) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status && !result.id,
           "native LRN error is not a policy rejection");
}
void rejected(const CudaDnnResult& result, CUresult status) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == status &&
               !result.original_status && !result.id,
           "pre-SDK rejection cannot invent original success");
}
std::array<std::uint64_t, 3> bits(const CudaDnnLrnState& value) {
    return {std::bit_cast<std::uint64_t>(value.alpha), std::bit_cast<std::uint64_t>(value.beta),
            std::bit_cast<std::uint64_t>(value.k)};
}
struct Object {
    bool is_live{};
    unsigned int destroys{};
    CudaDnnLrnState state{};
};
struct Fixture {
    std::array<Object, 8> objects{};
    unsigned int creates{}, sets{}, destroys{}, fail_destroy_call{};
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, set_status{CUDNN_STATUS_SUCCESS},
        destroy_status{CUDNN_STATUS_SUCCESS};
    bool has_partial_create{}, has_null_success{};
    void* duplicate{};
    std::uint32_t last_n{};
    std::array<std::uint64_t, 3> last_bits{};
} fixture;
void reset() {
    sdk::reset();
    fixture = {};
}
Object& object(CudaDnnLrn native) {
    sdk::owner();
    for (auto& candidate : fixture.objects)
        if (static_cast<void*>(&candidate) == native) {
            expect(candidate.is_live, "LRN fixture native descriptor is live");
            return candidate;
        }
    throw std::runtime_error("unknown native LRN descriptor");
}
CudaDnnDispatch dispatch() {
    auto result = sdk::dispatch();
    result.lrn_create = [](CudaDnnLrn* output) {
        sdk::owner();
        ++fixture.creates;
        if (fixture.has_null_success ||
            (fixture.create_status != CUDNN_STATUS_SUCCESS && !fixture.has_partial_create))
            return fixture.create_status;
        if (fixture.duplicate) {
            *output = static_cast<CudaDnnLrn>(fixture.duplicate);
            return fixture.create_status;
        }
        for (auto& candidate : fixture.objects)
            if (!candidate.is_live) {
                candidate = {};
                candidate.is_live = true;
                // These are original SDK defaults, never registry-owned state.
                candidate.state = {5, 1e-4, 0.75, 2.0};
                *output = reinterpret_cast<CudaDnnLrn>(&candidate);
                return fixture.create_status;
            }
        throw std::runtime_error("LRN fixture native capacity");
    };
    result.lrn_destroy = [](CudaDnnLrn native) {
        auto& candidate = object(native);
        expect(++candidate.destroys == 1, "uncertain native destruction is never retried");
        ++fixture.destroys;
        const auto status = fixture.fail_destroy_call == fixture.destroys
                                ? CUDNN_STATUS_INTERNAL_ERROR
                                : fixture.destroy_status;
        if (status == CUDNN_STATUS_SUCCESS) candidate.is_live = false;
        return status;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native cuDNN ABI.
    result.lrn_set = [](CudaDnnLrn native, unsigned int n, double alpha, double beta, double k) {
        auto& candidate = object(native);
        ++fixture.sets;
        const CudaDnnLrnState value{n, alpha, beta, k};
        fixture.last_n = n;
        fixture.last_bits = bits(value);
        if (fixture.set_status != CUDNN_STATUS_SUCCESS) return fixture.set_status;
        if (n < CUDNN_LRN_MIN_N || n > CUDNN_LRN_MAX_N) return CUDNN_STATUS_BAD_PARAM;
        candidate.state = value;
        return CUDNN_STATUS_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return result;
}
CudaDnnLrnState value() {
    return {3, 0.125, 0.5, 2.25};
}
void native_defaults_and_exact_setters() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto lrn = registry.create(Kind::kLrn);
    success(lrn);
    expect(lrn.id && fixture.creates == 1 && fixture.sets == 0,
           "creation never emulates native LRN defaults with a hidden setter");
    {
        const auto view = registry.borrow(lrn.id, Kind::kLrn);
        expect(view && view->kind == Kind::kLrn, "new descriptor returns a typed native lease");
        const auto& native = object(static_cast<CudaDnnLrn>(view->native)).state;
        expect(native.n == 5 && bits(native) == bits({5, 1e-4, 0.75, 2.0}),
               "original creation supplies untouched default parameters");
    }
    for (const std::uint32_t n : {1U, 3U, 4U, 16U}) {
        auto setting = value();
        setting.n = n;
        success(registry.set_lrn(lrn.id, setting));
        const auto view = registry.borrow(lrn.id, Kind::kLrn);
        const auto& native = object(static_cast<CudaDnnLrn>(view->native)).state;
        expect(native.n == n && bits(native) == bits(setting) && fixture.last_n == n &&
                   fixture.last_bits == bits(setting),
               "odd/even window and three native double values round-trip without narrowing");
    }
    for (const std::uint32_t n : {0U, 17U, 0x80000000U, UINT32_MAX}) {
        auto setting = value();
        setting.n = n;
        original(registry.set_lrn(lrn.id, setting), CUDNN_STATUS_BAD_PARAM);
        expect(fixture.last_n == n, "entire unsigned window reaches the native error path");
    }
    const std::array patterns{
        std::array<std::uint64_t, 3>{0x8000000000000000ULL, 1ULL, 0x7ff0000000000000ULL},
        std::array<std::uint64_t, 3>{0x7ff8000000000123ULL, 0xfff0000000000000ULL,
                                     0x7ff0000000000042ULL},
        std::array<std::uint64_t, 3>{0x000fffffffffffffULL, 0x7fefffffffffffffULL,
                                     0xfff8000000000321ULL}};
    fixture.set_status = CUDNN_STATUS_BAD_PARAM;
    for (const auto& pattern : patterns) {
        const CudaDnnLrnState setting{3, std::bit_cast<double>(pattern[0]),
                                      std::bit_cast<double>(pattern[1]),
                                      std::bit_cast<double>(pattern[2])};
        original(registry.set_lrn(lrn.id, setting), fixture.set_status);
        expect(fixture.last_bits == pattern,
               "zero/subnormal/infinity/NaN payload bits reach the native call unchanged");
    }
    fixture.set_status = CUDNN_STATUS_SUCCESS;
    success(registry.set_lrn(lrn.id, value()));
    success(registry.destroy(lrn.id));
    expect(fixture.destroys == 1, "explicit LRN destruction runs once");
    rejected(registry.set_lrn(lrn.id, value()), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.destroy(lrn.id), CUDA_ERROR_INVALID_HANDLE);
    success(registry.shutdown());
}
void bounds_and_optional_callbacks() {
    reset();
    {
        CudaDnnRegistry old(sdk::dispatch());
        rejected(old.create(Kind::kLrn), CUDA_ERROR_NOT_SUPPORTED);
        success(old.shutdown());
    }
    for (bool is_create_missing : {false, true}) {
        auto callbacks = dispatch();
        if (is_create_missing)
            callbacks.lrn_create = nullptr;
        else
            callbacks.lrn_destroy = nullptr;
        CudaDnnRegistry incomplete(callbacks);
        rejected(incomplete.create(Kind::kLrn), CUDA_ERROR_NOT_SUPPORTED);
        expect(!fixture.creates, "missing creation or cleanup callback prevents native ownership");
    }
    auto callbacks = dispatch();
    callbacks.lrn_set = nullptr;
    {
        CudaDnnRegistry incomplete(callbacks);
        const auto lrn = incomplete.create(Kind::kLrn).id;
        rejected(incomplete.set_lrn(lrn, value()), CUDA_ERROR_NOT_SUPPORTED);
        expect(!fixture.sets, "absent setter cannot create an emulated descriptor");
        success(incomplete.shutdown());
    }
    reset();
    CudaDnnRegistry registry(dispatch(), 2);
    fixture.create_status = CUDNN_STATUS_ALLOC_FAILED;
    original(registry.create(Kind::kLrn), fixture.create_status);
    expect(!registry.snapshot().objects, "known empty creation failure consumes no object");
    fixture.create_status = CUDNN_STATUS_SUCCESS;
    const auto lrn = registry.create(Kind::kLrn).id;
    const auto tensor = registry.create(Kind::kTensor).id;
    const auto creates = fixture.creates;
    rejected(registry.create(Kind::kLrn), CUDA_ERROR_OUT_OF_MEMORY);
    expect(fixture.creates == creates, "object bound is checked before SDK creation");
    rejected(registry.set_lrn(tensor, value()), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.set_lrn(999, value()), CUDA_ERROR_INVALID_HANDLE);
    expect(!fixture.sets && !registry.borrow(lrn, Kind::kTensor) &&
               !registry.borrow(tensor, Kind::kLrn),
           "foreign ID or kind cannot enter native setter or produce wrong-kind lease");
    success(registry.shutdown());
    expect(fixture.destroys == 1 && sdk::state.destroys == 1,
           "mixed tensor/LRN cleanup uses each exact native callback once");
}
void leases_and_owner_boundaries() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto lrn = registry.create(Kind::kLrn).id;
    auto lease = registry.borrow(lrn, Kind::kLrn);
    expect(lease && registry.set_lrn(lrn, value()).disposition == CudaDnnDisposition::kDeferred &&
               registry.destroy(lrn).disposition == CudaDnnDisposition::kDeferred &&
               !fixture.sets && !fixture.destroys,
           "queued immutable consumers defer mutation/destruction without touching native state");
    rejected(std::async(std::launch::async, [&] { return registry.set_lrn(lrn, value()); }).get(),
             CUDA_ERROR_UNKNOWN);
    const auto child = ::fork();
    expect(child >= 0, "fork LRN owner boundary control");
    if (!child)
        ::_exit(registry.set_lrn(lrn, value()).status == CUDA_ERROR_UNKNOWN &&
                        !registry.borrow(lrn, Kind::kLrn)
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               !fixture.sets,
           "foreign thread and inherited process never invoke native LRN SDK");
    registry.close();
    rejected(registry.set_lrn(lrn, value()), CUDA_ERROR_DEINITIALIZED);
    expect(!registry.borrow(lrn, Kind::kLrn) &&
               registry.shutdown().disposition == CudaDnnDisposition::kDeferred,
           "close rejects new work and waits for existing immutable consumer");
    lease.reset();
    success(registry.shutdown());
    success(registry.shutdown());
    expect(fixture.destroys == 1, "quiescent LRN shutdown is idempotent");
}
void isolated_ownership_failures() {
    for (unsigned int mode = 0; mode != 8; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "fork intentional LRN uncertainty");
        if (!child) {
            try {
                reset();
                auto registry = std::make_unique<CudaDnnRegistry>(dispatch());
                const auto lrn = registry->create(Kind::kLrn).id;
                if (mode <= 4) {
                    fixture.has_null_success = mode == 0;
                    fixture.has_partial_create = mode == 1 || mode == 4;
                    if (fixture.has_partial_create)
                        fixture.create_status = CUDNN_STATUS_INTERNAL_ERROR;
                    if (mode >= 2) {
                        const auto id = mode == 3 ? registry->create(Kind::kTensor).id : lrn;
                        const auto view =
                            registry->borrow(id, mode == 3 ? Kind::kTensor : Kind::kLrn);
                        fixture.duplicate = view->native;
                    }
                    const auto result = registry->create(Kind::kLrn);
                    expect(result.disposition == CudaDnnDisposition::kRejected && !result.id &&
                               result.original_status == fixture.create_status,
                           "null/partial/same-kind/cross-kind duplicate creation publishes no ID");
                } else if (mode == 5) {
                    fixture.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
                    original(registry->destroy(lrn), fixture.destroy_status);
                } else if (mode == 6) {
                    fixture.set_status = CUDNN_STATUS_EXECUTION_FAILED;
                    original(registry->set_lrn(lrn, value()), fixture.set_status);
                } else {
                    success(registry->create(Kind::kLrn));
                    fixture.fail_destroy_call = 2;
                    original(registry->shutdown(), CUDNN_STATUS_INTERNAL_ERROR);
                    expect(registry->snapshot().objects == 1 && fixture.destroys == 2,
                           "partial shutdown removes only already destroyed owners");
                }
                expect(registry->snapshot().is_quarantined, "LRN uncertain ownership is explicit");
                const auto creates = fixture.creates, sets = fixture.sets,
                           destroys = fixture.destroys;
                rejected(registry->create(Kind::kLrn), CUDA_ERROR_UNKNOWN);
                rejected(registry->set_lrn(lrn, value()), CUDA_ERROR_UNKNOWN);
                rejected(registry->shutdown(), CUDA_ERROR_UNKNOWN);
                registry.reset();
                expect(fixture.creates == creates && fixture.sets == sets &&
                           fixture.destroys == destroys,
                       "unknown ownership or mutation is never retried during teardown");
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "LRN ownership mode " << mode << ": " << error.what() << '\n';
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated LRN uncertain-ownership controls pass");
    }
}
}  // namespace
int main() {
    try {
        native_defaults_and_exact_setters();
        bounds_and_optional_callbacks();
        leases_and_owner_boundaries();
        isolated_ownership_failures();
        std::cout << "cuDNN LRN registry ownership checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
