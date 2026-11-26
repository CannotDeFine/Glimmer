#include "dnn_fixture.h"

#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <string_view>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace sdk = glimmer::tests::dnn;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "completion preserves the original cuDNN status");
}
void original(const CudaDnnResult& result, cudnnStatus_t status) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status && result.id == 0,
           "original failure is distinct from backend rejection");
}
void rejected(const CudaDnnResult& result, CUresult status) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == status &&
               !result.original_status && result.id == 0,
           "pre-SDK rejection does not invent an original status or handle");
}
CudaDnnTensor tensor() {
    return {.type = CUDNN_DATA_DOUBLE,
            .format = CUDNN_TENSOR_NCHW,
            .dimensions = 4,
            .sizes = {2, 3, 5, 7},
            .strides = {131, 41, 8, 1}};
}
void descriptor_arguments_and_queries() {
    sdk::reset();
    CudaDnnRegistry registry(sdk::dispatch());
    const auto created = registry.create(CudaDnnObjectKind::kTensor);
    success(created);
    const auto id = created.id;
    auto input = tensor();
    success(registry.set_tensor(id, CudaDnnTensorSetter::kFour, input));
    auto result = registry.get_tensor(id, 4, true);
    success(result);
    expect(result.tensor.type == input.type && result.tensor.sizes == input.sizes &&
               result.tensor.strides[0] == 105 && result.tensor.strides[1] == 35 &&
               result.tensor.strides[2] == 7 && result.tensor.strides[3] == 1,
           "4D getter returns actual original packed strides, not caller guesses");
    // The strided ABI has no format argument. Unused structure fields cannot
    // create a rejection or be converted to an out-of-range C++ enum.
    input.format = std::numeric_limits<std::uint32_t>::max();
    success(registry.set_tensor(id, CudaDnnTensorSetter::kFourStrided, input));
    result = registry.get_tensor(id, 4, true);
    success(result);
    expect(result.tensor.strides == input.strides, "4DEx preserves every explicit stride");
    input.dimensions = 5;
    input.sizes[4] = 11;
    input.strides[4] = 3;
    success(registry.set_tensor(id, CudaDnnTensorSetter::kNd, input));
    result = registry.get_tensor(id, CUDNN_DIM_MAX);
    success(result);
    expect(result.tensor.dimensions == 5 && result.tensor.sizes == input.sizes &&
               result.tensor.strides == input.strides,
           "Nd descriptor arrays retain their complete native values");
    result = registry.get_tensor(id, 2);
    success(result);
    expect(result.tensor.dimensions == 5 && result.tensor.sizes[0] == 2 &&
               result.tensor.sizes[1] == 3 && result.tensor.sizes[2] == 0 &&
               result.tensor.strides[2] == 0 && !registry.snapshot().is_quarantined,
           "a partial Nd query returns the full rank and only requested array entries");
    result = registry.get_tensor(id, 0);
    success(result);
    expect(result.tensor.dimensions == 5 && result.tensor.sizes == std::array<int, CUDNN_DIM_MAX>{},
           "zero-entry Nd query still preserves the original rank");
    input.format = CUDNN_TENSOR_NCHW;
    input.type = CUDNN_DATA_HALF;
    success(registry.set_tensor(id, CudaDnnTensorSetter::kNdPacked, input));
    result = registry.get_tensor(id, 5);
    success(result);
    expect(result.tensor.type == CUDNN_DATA_HALF && result.tensor.strides[0] == 1155 &&
               result.tensor.strides[4] == 1,
           "NdEx dispatch passes format/type/all dimensions and returns vendor strides");
    expect(sdk::state.tensor_setters == std::array<unsigned int, 4>{1, 1, 1, 1},
           "each public setter uses its distinct original SDK entry");
    auto view = registry.borrow(id, CudaDnnObjectKind::kTensor);
    expect(view.has_value(), "owned descriptor may be borrowed");
    sdk::object(view->native, CudaDnnObjectKind::kTensor).bytes = (std::size_t{1} << 34) + 17;
    result = registry.tensor_size(id);
    success(result);
    expect(result.value == (std::uint64_t{1} << 34) + 17,
           "native size_t result is not truncated or reconstructed from dimensions");
    view.reset();
    success(registry.destroy(id));
    rejected(registry.get_tensor(id, 4), CUDA_ERROR_INVALID_HANDLE);
    expect(sdk::state.destroys == 1, "one successful native destruction");
}
void activation_arguments_and_original_errors() {
    sdk::reset();
    CudaDnnRegistry registry(sdk::dispatch());
    const auto created = registry.create(CudaDnnObjectKind::kActivation);
    success(created);
    const auto id = created.id;
    original(registry.get_activation(id), CUDNN_STATUS_BAD_PARAM);
    CudaDnnActivationState input{CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 6.125};
    success(registry.set_activation(id, input));
    auto result = registry.get_activation(id);
    success(result);
    expect(result.activation.mode == input.mode && result.activation.nan == input.nan &&
               result.activation.coefficient == input.coefficient,
           "activation mode/NaN policy/double coefficient retain original values");
    input.coefficient = std::numeric_limits<double>::quiet_NaN();
    success(registry.set_activation(id, input));
    expect(std::isnan(registry.get_activation(id).activation.coefficient),
           "opaque double coefficient is delegated unchanged to native semantics");
    input.mode = 7;  // Representable but not a declared cuDNN activation mode.
    original(registry.set_activation(id, input), CUDNN_STATUS_NOT_SUPPORTED);
    expect(!registry.snapshot().is_quarantined, "known native rejection preserves usable metadata");
    const auto sets = sdk::state.sets;
    input.mode = 8;
    rejected(registry.set_activation(id, input), CUDA_ERROR_INVALID_VALUE);
    input.mode = CUDNN_ACTIVATION_RELU;
    input.nan = 2;
    rejected(registry.set_activation(id, input), CUDA_ERROR_INVALID_VALUE);
    expect(sdk::state.sets == sets, "unrepresentable activation enum values never enter SDK code");
    sdk::state.get_status = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
    result = registry.get_activation(id);
    original(result, sdk::state.get_status);
    expect(result.activation.mode == 0 && result.activation.nan == 0 &&
               result.activation.coefficient == 0,
           "even partially touched failed native outputs are not published");
    success(registry.destroy(id));
}
void tensor_bounds_and_failed_outputs() {
    sdk::reset();
    CudaDnnRegistry registry(sdk::dispatch());
    const auto created = registry.create(CudaDnnObjectKind::kTensor);
    success(created);
    const auto id = created.id;
    auto input = tensor();
    success(registry.set_tensor(id, CudaDnnTensorSetter::kNd, input));
    input.type = 31;
    original(registry.set_tensor(id, CudaDnnTensorSetter::kNd, input),
             CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    input = tensor();
    input.format = 3;
    original(registry.set_tensor(id, CudaDnnTensorSetter::kFour, input),
             CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    input = tensor();
    input.sizes[0] = -1;
    original(registry.set_tensor(id, CudaDnnTensorSetter::kNd, input), CUDNN_STATUS_BAD_PARAM);
    input = tensor();
    input.dimensions = 0;
    original(registry.set_tensor(id, CudaDnnTensorSetter::kNd, input), CUDNN_STATUS_BAD_PARAM);
    const auto calls = sdk::state.sets;
    input = tensor();
    input.type = 32;
    rejected(registry.set_tensor(id, CudaDnnTensorSetter::kNd, input), CUDA_ERROR_INVALID_VALUE);
    input = tensor();
    input.format = 4;
    rejected(registry.set_tensor(id, CudaDnnTensorSetter::kFour, input), CUDA_ERROR_INVALID_VALUE);
    input = tensor();
    input.dimensions = -1;
    rejected(registry.set_tensor(id, CudaDnnTensorSetter::kNd, input), CUDA_ERROR_INVALID_VALUE);
    input.dimensions = CUDNN_DIM_MAX + 1;
    rejected(registry.set_tensor(id, CudaDnnTensorSetter::kNd, input), CUDA_ERROR_INVALID_VALUE);
    input.dimensions = 3;
    rejected(registry.set_tensor(id, CudaDnnTensorSetter::kFourStrided, input),
             CUDA_ERROR_INVALID_VALUE);
    input = tensor();
    rejected(registry.set_tensor(id, static_cast<CudaDnnTensorSetter>(999), input),
             CUDA_ERROR_INVALID_VALUE);
    expect(sdk::state.sets == calls, "wire enum/rank/setter bounds reject before native calls");
    const auto queries = sdk::state.gets;
    rejected(registry.get_tensor(id, -1), CUDA_ERROR_INVALID_VALUE);
    rejected(registry.get_tensor(id, CUDNN_DIM_MAX + 1), CUDA_ERROR_INVALID_VALUE);
    rejected(registry.get_tensor(id, 3, true), CUDA_ERROR_INVALID_VALUE);
    expect(sdk::state.gets == queries, "bad query buffers never reach the SDK");
    sdk::state.get_status = CUDNN_STATUS_BAD_PARAM;
    auto result = registry.get_tensor(id, 4, true);
    original(result, sdk::state.get_status);
    expect(result.tensor.dimensions == 0 && result.tensor.type == 0 &&
               result.tensor.sizes == std::array<int, CUDNN_DIM_MAX>{},
           "failed 4D query cannot publish partially touched outputs");
    result = registry.get_tensor(id, 4);
    original(result, sdk::state.get_status);
    expect(result.tensor.dimensions == 0, "failed Nd query cannot publish its touched rank");
    sdk::state.size_status = CUDNN_STATUS_NOT_SUPPORTED;
    result = registry.tensor_size(id);
    original(result, sdk::state.size_status);
    expect(result.value == 0, "failed size query cannot publish partially touched bytes");
    rejected(registry.get_activation(id), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.set_activation(id, {}), CUDA_ERROR_INVALID_HANDLE);
    success(registry.shutdown());
}
void bounds_and_nonrecycled_ids() {
    sdk::reset();
    bool did_throw{};
    try {
        CudaDnnRegistry registry(sdk::dispatch(), 0);
    } catch (const std::invalid_argument&) {
        did_throw = true;
    }
    expect(did_throw, "zero retention bound is rejected at construction");
    auto incomplete = sdk::dispatch();
    incomplete.tensor_get_nd = nullptr;
    did_throw = false;
    try {
        CudaDnnRegistry registry(incomplete);
    } catch (const std::invalid_argument&) {
        did_throw = true;
    }
    expect(did_throw && sdk::state.creates == 0, "incomplete dispatch cannot issue native work");
    CudaDnnRegistry registry(sdk::dispatch(), 2);
    rejected(registry.create(static_cast<CudaDnnObjectKind>(999)), CUDA_ERROR_INVALID_VALUE);
    sdk::state.create_status = CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED;
    original(registry.create(CudaDnnObjectKind::kTensor), sdk::state.create_status);
    expect(registry.snapshot().objects == 0 && !registry.snapshot().is_quarantined,
           "known-empty native creation failure publishes and retains no resource");
    sdk::state.create_status = CUDNN_STATUS_SUCCESS;
    const auto first = registry.create(CudaDnnObjectKind::kTensor);
    const auto second = registry.create(CudaDnnObjectKind::kActivation);
    success(first);
    success(second);
    const auto calls = sdk::state.creates;
    rejected(registry.create(CudaDnnObjectKind::kTensor), CUDA_ERROR_OUT_OF_MEMORY);
    expect(sdk::state.creates == calls, "object capacity is secured before native creation");
    rejected(registry.tensor_size(second.id), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.set_tensor(second.id, CudaDnnTensorSetter::kNd, tensor()),
             CUDA_ERROR_INVALID_HANDLE);
    success(registry.destroy(first.id));
    const auto third = registry.create(CudaDnnObjectKind::kTensor);
    success(third);
    expect(third.id > second.id && second.id > first.id,
           "recycled native address never reuses an opaque resource identity");
    rejected(registry.destroy(first.id), CUDA_ERROR_INVALID_HANDLE);
    expect(!registry.borrow(first.id, CudaDnnObjectKind::kTensor) &&
               !registry.borrow(third.id, CudaDnnObjectKind::kActivation),
           "stale and cross-kind borrows reject");
    success(registry.shutdown());
    success(registry.shutdown());
    expect(sdk::state.destroys == 3, "repeated shutdown cannot destroy successful records twice");
}
void immutable_borrows_and_close() {
    sdk::reset();
    CudaDnnRegistry registry(sdk::dispatch());
    const auto tensor_id = registry.create(CudaDnnObjectKind::kTensor).id;
    const auto activation_id = registry.create(CudaDnnObjectKind::kActivation).id;
    success(registry.set_tensor(tensor_id, CudaDnnTensorSetter::kNd, tensor()));
    success(registry.set_activation(activation_id, {CUDNN_ACTIVATION_RELU, 0, 0}));
    auto tensor_view = registry.borrow(tensor_id, CudaDnnObjectKind::kTensor);
    auto activation_view = registry.borrow(activation_id, CudaDnnObjectKind::kActivation);
    expect(tensor_view && activation_view && registry.snapshot().borrowed == 2,
           "each immutable native descriptor view retains ownership");
    const auto sets = sdk::state.sets;
    expect(registry.set_tensor(tensor_id, CudaDnnTensorSetter::kNd, tensor()).disposition ==
                   CudaDnnDisposition::kDeferred &&
               registry.set_activation(activation_id, {}).disposition ==
                   CudaDnnDisposition::kDeferred &&
               registry.destroy(tensor_id).disposition == CudaDnnDisposition::kDeferred &&
               registry.destroy(activation_id).disposition == CudaDnnDisposition::kDeferred &&
               sdk::state.sets == sets && sdk::state.destroys == 0,
           "borrowed descriptors cannot be mutated or destroyed before physical retirement");
    rejected(registry.set_tensor(tensor_id, static_cast<CudaDnnTensorSetter>(99), tensor()),
             CUDA_ERROR_INVALID_VALUE);
    success(registry.get_tensor(tensor_id, 4));
    success(registry.get_activation(activation_id));
    success(registry.tensor_size(tensor_id));
    // A lease may be released from a consumer without invoking an SDK function.
    auto release_elsewhere =
        std::async(std::launch::async, [view = std::move(tensor_view)]() mutable { view.reset(); });
    release_elsewhere.get();
    expect(sdk::state.destroys == 0 && registry.snapshot().borrowed == 1,
           "foreign lease release performs no vendor operation");
    success(registry.set_tensor(tensor_id, CudaDnnTensorSetter::kNd, tensor()));
    registry.close();
    expect(registry.snapshot().is_closing && !registry.snapshot().is_closed,
           "close records intent but does not destroy pending native objects");
    rejected(registry.create(CudaDnnObjectKind::kTensor), CUDA_ERROR_DEINITIALIZED);
    rejected(registry.destroy(tensor_id), CUDA_ERROR_DEINITIALIZED);
    rejected(registry.set_activation(activation_id, {}), CUDA_ERROR_DEINITIALIZED);
    expect(!registry.borrow(tensor_id, CudaDnnObjectKind::kTensor),
           "closing owner cannot admit a new physical consumer");
    success(registry.get_activation(activation_id));
    expect(registry.shutdown().disposition == CudaDnnDisposition::kDeferred &&
               sdk::state.destroys == 0,
           "shutdown waits for all borrowed descriptors before any native destruction");
    activation_view.reset();
    success(registry.shutdown());
    expect(registry.snapshot().is_closed && registry.snapshot().objects == 0 &&
               sdk::state.destroys == 2,
           "final retirement allows exactly one cleanup per native descriptor");
}
void owner_and_fork_isolation() {
    sdk::reset();
    CudaDnnRegistry registry(sdk::dispatch());
    const auto id = registry.create(CudaDnnObjectKind::kTensor).id;
    success(registry.set_tensor(id, CudaDnnTensorSetter::kNd, tensor()));
    const auto calls = sdk::state.creates + sdk::state.destroys + sdk::state.gets + sdk::state.sets;
    std::async(std::launch::async, [&] {
        rejected(registry.create(CudaDnnObjectKind::kTensor), CUDA_ERROR_UNKNOWN);
        rejected(registry.destroy(id), CUDA_ERROR_UNKNOWN);
        rejected(registry.get_tensor(id, 4), CUDA_ERROR_UNKNOWN);
        rejected(registry.set_tensor(id, CudaDnnTensorSetter::kNd, tensor()), CUDA_ERROR_UNKNOWN);
        rejected(registry.shutdown(), CUDA_ERROR_UNKNOWN);
        expect(
            !registry.borrow(id, CudaDnnObjectKind::kTensor) && registry.snapshot().is_quarantined,
            "foreign owner cannot observe or borrow retained native metadata");
        registry.close();
    }).get();
    expect(
        !registry.snapshot().is_closing &&
            sdk::state.creates + sdk::state.destroys + sdk::state.gets + sdk::state.sets == calls,
        "foreign methods neither mutate state nor invoke the injected SDK");
    const auto child = ::fork();
    expect(child >= 0, "fork isolation fixture");
    if (child == 0) {
        try {
            rejected(registry.get_tensor(id, 4), CUDA_ERROR_UNKNOWN);
            rejected(registry.shutdown(), CUDA_ERROR_UNKNOWN);
            expect(!registry.borrow(id, CudaDnnObjectKind::kTensor),
                   "inherited descriptor cannot escape its original process");
            ::_exit(0);
        } catch (...) {
            ::_exit(1);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "child rejected inherited ownership without native issue");
    success(registry.shutdown());
}

// Quarantine intentionally retains uncertain native resources. Isolated child
// processes keep those safety tests from becoming host-process leak reports.
template <typename Function>
void isolated(Function function) {
    const auto child = ::fork();
    expect(child >= 0, "isolated uncertainty fixture");
    if (child == 0) {
        try {
            function();
            ::_exit(0);
        } catch (...) {
            ::_exit(1);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated uncertain-ownership checks");
}
void uncertain_native_ownership() {
    for (unsigned int test_case = 0; test_case != 6; ++test_case)
        isolated([test_case] {
            const auto scenario = test_case % 3;
            const auto kind =
                test_case < 3 ? CudaDnnObjectKind::kTensor : CudaDnnObjectKind::kActivation;
            sdk::reset();
            {
                CudaDnnRegistry registry(sdk::dispatch());
                if (scenario == 0) {
                    sdk::state.create_status = CUDNN_STATUS_INTERNAL_ERROR;
                    sdk::state.partial_create = true;
                } else if (scenario == 1)
                    sdk::state.null_create = true;
                else {
                    success(registry.create(kind));
                    sdk::state.duplicate_create = true;
                }
                const auto result = registry.create(kind);
                expect(result.disposition == CudaDnnDisposition::kRejected &&
                           result.status == CUDA_ERROR_UNKNOWN && result.id == 0 &&
                           result.original_status == sdk::state.create_status &&
                           registry.snapshot().is_quarantined &&
                           registry.snapshot().objects == (scenario == 2 ? 2U : 1U),
                       "partial/null-success/duplicate creation retains uncertain ownership");
                rejected(registry.shutdown(), CUDA_ERROR_UNKNOWN);
                expect(sdk::state.destroys == 0,
                       "uncertain creations are never speculatively freed");
            }
            expect(sdk::state.destroys == 0, "destructor does not retry uncertain native cleanup");
        });
    isolated([] {
        sdk::reset();
        {
            CudaDnnRegistry registry(sdk::dispatch());
            const auto id = registry.create(CudaDnnObjectKind::kActivation).id;
            sdk::state.destroy_status = CUDNN_STATUS_BAD_PARAM;
            original(registry.destroy(id), sdk::state.destroy_status);
            expect(registry.snapshot().is_quarantined && registry.snapshot().objects == 1,
                   "even a known rejection during destruction leaves ownership uncertain");
            sdk::state.destroy_status = CUDNN_STATUS_SUCCESS;
            rejected(registry.destroy(id), CUDA_ERROR_UNKNOWN);
            rejected(registry.shutdown(), CUDA_ERROR_UNKNOWN);
        }
        expect(sdk::state.destroys == 1, "failed explicit native destruction is attempted once");
    });
    isolated([] {
        sdk::reset();
        {
            CudaDnnRegistry registry(sdk::dispatch());
            success(registry.create(CudaDnnObjectKind::kTensor));
            success(registry.create(CudaDnnObjectKind::kActivation));
            success(registry.create(CudaDnnObjectKind::kTensor));
            sdk::state.fail_destroy_call = 2;
            original(registry.shutdown(), CUDNN_STATUS_INTERNAL_ERROR);
            expect(registry.snapshot().objects == 2 && sdk::state.destroys == 2,
                   "successful early cleanup is erased before a later shutdown failure");
            rejected(registry.shutdown(), CUDA_ERROR_UNKNOWN);
        }
        expect(sdk::state.destroys == 2, "shutdown and destructor never repeat partial cleanup");
    });
    isolated([] {
        sdk::reset();
        {
            CudaDnnRegistry registry(sdk::dispatch());
            const auto id = registry.create(CudaDnnObjectKind::kTensor).id;
            sdk::state.set_status = CUDNN_STATUS_INTERNAL_ERROR;
            original(registry.set_tensor(id, CudaDnnTensorSetter::kNd, tensor()),
                     sdk::state.set_status);
            expect(registry.snapshot().is_quarantined &&
                       !registry.borrow(id, CudaDnnObjectKind::kTensor),
                   "uncertain mutation cannot feed a later physical consumer");
        }
        expect(sdk::state.destroys == 0, "uncertain mutation is retained without guessed cleanup");
    });
    isolated([] {
        sdk::reset();
        {
            CudaDnnRegistry registry(sdk::dispatch());
            const auto id = registry.create(CudaDnnObjectKind::kTensor).id;
            success(registry.set_tensor(id, CudaDnnTensorSetter::kNd, tensor()));
            sdk::state.reported_rank = CUDNN_DIM_MAX + 1;
            const auto result = registry.get_tensor(id, 2);
            expect(result.disposition == CudaDnnDisposition::kRejected &&
                       result.status == CUDA_ERROR_UNKNOWN &&
                       result.original_status == CUDNN_STATUS_SUCCESS &&
                       result.tensor.dimensions == 0 && registry.snapshot().is_quarantined,
                   "impossible successful rank quarantines ownership without losing provenance");
        }
        expect(sdk::state.destroys == 0, "corrupt successful native output is never retried");
    });
    for (unsigned int scenario = 0; scenario != 3; ++scenario)
        isolated([scenario] {
            sdk::reset();
            {
                CudaDnnRegistry registry(sdk::dispatch());
                const auto kind =
                    scenario == 0 ? CudaDnnObjectKind::kTensor : CudaDnnObjectKind::kActivation;
                const auto id = registry.create(kind).id;
                if (scenario == 0) {
                    success(registry.set_tensor(id, CudaDnnTensorSetter::kNd, tensor()));
                    sdk::state.reported_type = std::numeric_limits<std::uint32_t>::max();
                } else {
                    success(registry.set_activation(id, {CUDNN_ACTIVATION_RELU, 0, 2.5}));
                    if (scenario == 1)
                        sdk::state.reported_mode = 8;
                    else
                        sdk::state.reported_nan = 2;
                }
                const auto result =
                    scenario == 0 ? registry.get_tensor(id, 4) : registry.get_activation(id);
                expect(result.disposition == CudaDnnDisposition::kRejected &&
                           result.status == CUDA_ERROR_UNKNOWN &&
                           result.original_status == CUDNN_STATUS_SUCCESS &&
                           result.tensor.type == 0 && result.tensor.dimensions == 0 &&
                           result.activation.mode == 0 && result.activation.nan == 0 &&
                           result.activation.coefficient == 0 && registry.snapshot().is_quarantined,
                       "malformed native enum bits reject before enum loads or output publication");
            }
            expect(sdk::state.destroys == 0, "malformed-success quarantine never guesses cleanup");
        });
    isolated([] {
        sdk::reset();
        std::optional<CudaDnnObjectView> view;
        {
            CudaDnnRegistry registry(sdk::dispatch());
            const auto id = registry.create(CudaDnnObjectKind::kTensor).id;
            view = registry.borrow(id, CudaDnnObjectKind::kTensor);
        }
        expect(view && sdk::state.destroys == 0,
               "destructor cannot invalidate a descriptor still owned by physical work");
        view.reset();
        expect(sdk::state.destroys == 0, "late lease release is not deferred SDK destruction");
    });
    isolated([] {
        auto registry = std::async(std::launch::async, [] {
                            sdk::reset();
                            auto owned = std::make_unique<CudaDnnRegistry>(sdk::dispatch());
                            success(owned->create(CudaDnnObjectKind::kTensor));
                            return owned;
                        }).get();
        // The creating thread has exited. Even recycling its numeric thread
        // identifier cannot recreate the original thread-local owner token.
        std::async(std::launch::async, [&] {
            rejected(registry->shutdown(), CUDA_ERROR_UNKNOWN);
            rejected(registry->create(CudaDnnObjectKind::kTensor), CUDA_ERROR_UNKNOWN);
        }).get();
        registry.reset();
        expect(sdk::state.destroys == 0,
               "departed thread ownership cannot migrate or trigger foreign native cleanup");
    });
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--quarantine")
            uncertain_native_ownership();
        else {
            expect(argc == 1, "unsupported cuDNN registry test argument");
            descriptor_arguments_and_queries();
            activation_arguments_and_original_errors();
            tensor_bounds_and_failed_outputs();
            bounds_and_nonrecycled_ids();
            immutable_borrows_and_close();
            owner_and_fork_isolation();
        }
        std::cout << "cuDNN descriptor ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
