#include "dnn_fixture.h"

#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace sdk = glimmer::tests::dnn;
using Kind = CudaDnnObjectKind;
using State = CudaDnnReductionState;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "native reduction success retains both status domains");
}
void original(const CudaDnnResult& result, cudnnStatus_t status) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status && !result.id,
           "native reduction error is not a policy rejection");
}
void rejected(const CudaDnnResult& result, CUresult status) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == status &&
               !result.original_status && !result.id,
           "pre-SDK rejection cannot invent original success");
}
bool equal(const State& a, const State& b) {
    return a.operation == b.operation && a.type == b.type && a.nan == b.nan &&
           a.indices == b.indices && a.indices_type == b.indices_type;
}
struct Object {
    bool is_live{};
    unsigned int destroys{};
    State state{};
};
struct Fixture {
    std::array<Object, 8> objects{};
    unsigned int creates{}, sets{}, destroys{}, fail_destroy_call{};
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, set_status{CUDNN_STATUS_SUCCESS},
        destroy_status{CUDNN_STATUS_SUCCESS};
    bool has_partial_create{}, has_null_success{};
    void* duplicate{};
    State last{};
} fixture;
// Deliberately injected creation state, not an assertion about vendor defaults.
constexpr State native_defaults{CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_DOUBLE, CUDNN_PROPAGATE_NAN,
                                CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES};
void reset() {
    sdk::reset();
    fixture = {};
}
Object& object(CudaDnnReduction native) {
    sdk::owner();
    for (auto& candidate : fixture.objects)
        if (static_cast<void*>(&candidate) == native) {
            expect(candidate.is_live, "reduction fixture native descriptor is live");
            return candidate;
        }
    throw std::runtime_error("unknown native reduction descriptor");
}
CudaDnnDispatch dispatch() {
    auto result = sdk::dispatch();
    result.reduction_create = [](CudaDnnReduction* output) {
        sdk::owner();
        ++fixture.creates;
        if (fixture.has_null_success ||
            (fixture.create_status != CUDNN_STATUS_SUCCESS && !fixture.has_partial_create))
            return fixture.create_status;
        if (fixture.duplicate) {
            *output = static_cast<CudaDnnReduction>(fixture.duplicate);
            return fixture.create_status;
        }
        for (auto& candidate : fixture.objects)
            if (!candidate.is_live) {
                candidate = {};
                candidate.is_live = true;
                candidate.state = native_defaults;
                *output = reinterpret_cast<CudaDnnReduction>(&candidate);
                return fixture.create_status;
            }
        throw std::runtime_error("reduction fixture native capacity");
    };
    result.reduction_destroy = [](CudaDnnReduction native) {
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
    result.reduction_set = [](CudaDnnReduction native, cudnnReduceTensorOp_t operation,
                              cudnnDataType_t type, cudnnNanPropagation_t nan,
                              decltype(CUDNN_REDUCE_TENSOR_NO_INDICES) indices,
                              decltype(CUDNN_32BIT_INDICES) indices_type) {
        auto& candidate = object(native);
        ++fixture.sets;
        fixture.last = {static_cast<std::uint32_t>(operation), static_cast<std::uint32_t>(type),
                        static_cast<std::uint32_t>(nan), static_cast<std::uint32_t>(indices),
                        static_cast<std::uint32_t>(indices_type)};
        if (fixture.set_status != CUDNN_STATUS_SUCCESS) return fixture.set_status;
        candidate.state = fixture.last;
        return CUDNN_STATUS_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return result;
}
State value() {
    return {CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
            CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES};
}
void native_defaults_and_exact_setters() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto reduction = registry.create(Kind::kReduction);
    success(reduction);
    expect(reduction.id && fixture.creates == 1 && fixture.sets == 0,
           "creation never emulates native reduction defaults with a hidden setter");
    {
        const auto view = registry.borrow(reduction.id, Kind::kReduction);
        expect(view && view->kind == Kind::kReduction,
               "new descriptor returns a typed native lease");
        expect(equal(object(static_cast<CudaDnnReduction>(view->native)).state, native_defaults),
               "original creation supplies untouched descriptor parameters");
    }
    for (std::uint32_t operation = 0; operation != 9; ++operation) {
        auto setting = value();
        setting.operation = operation;
        success(registry.set_reduction(reduction.id, setting));
        const auto view = registry.borrow(reduction.id, Kind::kReduction);
        expect(equal(object(static_cast<CudaDnnReduction>(view->native)).state, setting) &&
                   equal(fixture.last, setting),
               "all named reduction operations reach their original setter unchanged");
    }
    fixture.set_status = CUDNN_STATUS_NOT_SUPPORTED;
    for (std::uint32_t operation = 9; operation != 16; ++operation) {
        auto setting = value();
        setting.operation = operation;
        original(registry.set_reduction(reduction.id, setting), fixture.set_status);
        expect(equal(fixture.last, setting),
               "representable unnamed operations reach the native error path");
    }
    const std::array<std::uint32_t State::*, 4> fields{&State::type, &State::nan, &State::indices,
                                                       &State::indices_type};
    const std::array<std::uint32_t, 4> maxima{31, 1, 1, 3};
    for (std::size_t field = 0; field != fields.size(); ++field) {
        for (std::uint32_t raw = 0; raw <= maxima[field]; ++raw) {
            auto setting = value();
            setting.*fields[field] = raw;
            original(registry.set_reduction(reduction.id, setting), fixture.set_status);
            expect(equal(fixture.last, setting),
                   "all representable enum values reach original validation without narrowing");
        }
    }
    fixture.set_status = CUDNN_STATUS_BAD_PARAM;
    original(registry.set_reduction(reduction.id, value()), fixture.set_status);
    expect(!registry.snapshot().is_quarantined, "known native rejections retain proven ownership");
    fixture.set_status = CUDNN_STATUS_SUCCESS;
    success(registry.set_reduction(reduction.id, value()));
    success(registry.destroy(reduction.id));
    expect(fixture.destroys == 1, "explicit reduction destruction runs once");
    rejected(registry.set_reduction(reduction.id, value()), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.destroy(reduction.id), CUDA_ERROR_INVALID_HANDLE);
    success(registry.shutdown());
}
void bounds_and_optional_callbacks() {
    reset();
    {
        CudaDnnRegistry old(sdk::dispatch());
        rejected(old.create(Kind::kReduction), CUDA_ERROR_NOT_SUPPORTED);
        success(old.shutdown());
    }
    for (bool is_create_missing : {false, true}) {
        auto callbacks = dispatch();
        if (is_create_missing)
            callbacks.reduction_create = nullptr;
        else
            callbacks.reduction_destroy = nullptr;
        CudaDnnRegistry incomplete(callbacks);
        rejected(incomplete.create(Kind::kReduction), CUDA_ERROR_NOT_SUPPORTED);
        expect(!fixture.creates, "missing create or destroy prevents native ownership");
    }
    auto callbacks = dispatch();
    callbacks.reduction_set = nullptr;
    {
        CudaDnnRegistry incomplete(callbacks);
        const auto reduction = incomplete.create(Kind::kReduction).id;
        rejected(incomplete.set_reduction(reduction, value()), CUDA_ERROR_NOT_SUPPORTED);
        expect(!fixture.sets, "absent setter cannot create an emulated descriptor");
        success(incomplete.shutdown());
    }
    reset();
    CudaDnnRegistry registry(dispatch(), 2);
    fixture.create_status = CUDNN_STATUS_ALLOC_FAILED;
    original(registry.create(Kind::kReduction), fixture.create_status);
    expect(!registry.snapshot().objects, "known empty creation failure consumes no object");
    fixture.create_status = CUDNN_STATUS_SUCCESS;
    const auto reduction = registry.create(Kind::kReduction).id;
    const auto tensor = registry.create(Kind::kTensor).id;
    const auto creates = fixture.creates;
    rejected(registry.create(Kind::kReduction), CUDA_ERROR_OUT_OF_MEMORY);
    expect(fixture.creates == creates, "object bound is checked before SDK creation");
    rejected(registry.set_reduction(tensor, value()), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.set_reduction(999, value()), CUDA_ERROR_INVALID_HANDLE);
    expect(!fixture.sets && !registry.borrow(reduction, Kind::kTensor) &&
               !registry.borrow(tensor, Kind::kReduction),
           "foreign ID or kind cannot enter native setter or produce wrong-kind lease");
    const std::array<std::uint32_t State::*, 5> fields{&State::operation, &State::type, &State::nan,
                                                       &State::indices, &State::indices_type};
    const std::array<std::uint32_t, 5> maxima{15, 31, 1, 1, 3};
    for (std::size_t field = 0; field != fields.size(); ++field) {
        for (const auto raw : {maxima[field] + 1, 0x80000000U, UINT32_MAX}) {
            auto setting = value();
            setting.*fields[field] = raw;
            rejected(registry.set_reduction(reduction, setting), CUDA_ERROR_INVALID_VALUE);
        }
    }
    expect(!fixture.sets && !registry.snapshot().is_quarantined,
           "unrepresentable values reject before SDK enum conversion without quarantining");
    success(registry.shutdown());
    expect(fixture.destroys == 1 && sdk::state.destroys == 1,
           "mixed tensor/reduction cleanup uses each exact native callback once");
}
void leases_and_owner_boundaries() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto reduction = registry.create(Kind::kReduction).id;
    auto lease = registry.borrow(reduction, Kind::kReduction);
    expect(lease &&
               registry.set_reduction(reduction, value()).disposition ==
                   CudaDnnDisposition::kDeferred &&
               registry.destroy(reduction).disposition == CudaDnnDisposition::kDeferred &&
               !fixture.sets && !fixture.destroys,
           "immutable consumers defer native mutation/destruction");
    rejected(
        std::async(std::launch::async, [&] { return registry.set_reduction(reduction, value()); })
            .get(),
        CUDA_ERROR_UNKNOWN);
    const auto child = ::fork();
    expect(child >= 0, "fork reduction owner boundary control");
    if (!child)
        ::_exit(registry.set_reduction(reduction, value()).status == CUDA_ERROR_UNKNOWN &&
                        !registry.borrow(reduction, Kind::kReduction)
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               !fixture.sets,
           "foreign thread and inherited process never invoke native reduction SDK");
    lease.reset();
    success(registry.set_reduction(reduction, value()));
    lease = registry.borrow(reduction, Kind::kReduction);
    registry.close();
    rejected(registry.set_reduction(reduction, value()), CUDA_ERROR_DEINITIALIZED);
    rejected(registry.create(Kind::kReduction), CUDA_ERROR_DEINITIALIZED);
    expect(!registry.borrow(reduction, Kind::kReduction) &&
               registry.shutdown().disposition == CudaDnnDisposition::kDeferred,
           "close rejects new work and waits for existing immutable consumer");
    lease.reset();
    success(registry.shutdown());
    success(registry.shutdown());
    expect(fixture.destroys == 1, "quiescent reduction shutdown is idempotent");
}
void isolated_ownership_failures() {
    for (unsigned int mode = 0; mode != 8; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "fork intentional reduction uncertainty");
        if (!child) {
            try {
                reset();
                auto registry = std::make_unique<CudaDnnRegistry>(dispatch());
                const auto reduction = registry->create(Kind::kReduction).id;
                if (mode <= 4) {
                    fixture.has_null_success = mode == 0;
                    fixture.has_partial_create = mode == 1 || mode == 4;
                    if (fixture.has_partial_create)
                        fixture.create_status = CUDNN_STATUS_INTERNAL_ERROR;
                    if (mode >= 2) {
                        const auto id = mode == 3 ? registry->create(Kind::kTensor).id : reduction;
                        const auto view =
                            registry->borrow(id, mode == 3 ? Kind::kTensor : Kind::kReduction);
                        fixture.duplicate = view->native;
                    }
                    const auto result = registry->create(Kind::kReduction);
                    expect(result.disposition == CudaDnnDisposition::kRejected && !result.id &&
                               result.original_status == fixture.create_status,
                           "null/partial/same-kind/cross-kind duplicate creation publishes no ID");
                } else if (mode == 5) {
                    fixture.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
                    original(registry->destroy(reduction), fixture.destroy_status);
                } else if (mode == 6) {
                    fixture.set_status = CUDNN_STATUS_EXECUTION_FAILED;
                    original(registry->set_reduction(reduction, value()), fixture.set_status);
                } else {
                    success(registry->create(Kind::kReduction));
                    fixture.fail_destroy_call = 2;
                    original(registry->shutdown(), CUDNN_STATUS_INTERNAL_ERROR);
                    expect(registry->snapshot().objects == 1 && fixture.destroys == 2,
                           "partial shutdown removes only already destroyed owners");
                }
                expect(registry->snapshot().is_quarantined,
                       "reduction uncertain ownership is explicit");
                const auto creates = fixture.creates, sets = fixture.sets,
                           destroys = fixture.destroys;
                rejected(registry->create(Kind::kReduction), CUDA_ERROR_UNKNOWN);
                rejected(registry->set_reduction(reduction, value()), CUDA_ERROR_UNKNOWN);
                rejected(registry->shutdown(), CUDA_ERROR_UNKNOWN);
                registry.reset();
                expect(fixture.creates == creates && fixture.sets == sets &&
                           fixture.destroys == destroys,
                       "unknown ownership or mutation is never retried during teardown");
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "Reduction ownership mode " << mode << ": " << error.what() << '\n';
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated reduction uncertain-ownership controls pass");
    }
}
}  // namespace
int main() {
    try {
        native_defaults_and_exact_setters();
        bounds_and_optional_callbacks();
        leases_and_owner_boundaries();
        isolated_ownership_failures();
        std::cout << "cuDNN reduction registry ownership checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
