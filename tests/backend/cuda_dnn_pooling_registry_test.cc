#include "dnn_fixture.h"

#include <future>
#include <iostream>
#include <optional>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace sdk = glimmer::tests::dnn;
using Kind = CudaDnnObjectKind;
using PoolingMode = decltype(CUDNN_POOLING_MAX);
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "native success preserves both status domains");
}
void original(const CudaDnnResult& result, cudnnStatus_t status) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status && !result.id,
           "original failure is not a fabricated policy rejection");
}
void rejected(const CudaDnnResult& result, CUresult status) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == status &&
               !result.original_status && !result.id,
           "pre-SDK rejection has no fabricated original status");
}
struct Object {
    bool is_live{};
    unsigned int destroys{};
    CudaDnnPoolingState pooling;
};
struct Fixture {
    std::array<Object, 8> objects{};
    unsigned int creates{}, destroys{}, sets{}, gets{}, shapes{}, fail_destroy_call{};
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, destroy_status{CUDNN_STATUS_SUCCESS},
        set_status{CUDNN_STATUS_SUCCESS}, get_status{CUDNN_STATUS_SUCCESS},
        shape_status{CUDNN_STATUS_SUCCESS};
    bool has_partial_create{}, has_null_success{}, has_invalid_capacity_success{};
    bool has_invalid_shape_success{};
    void* duplicate{};
    std::optional<std::uint32_t> reported_mode, reported_nan;
    std::optional<int> reported_rank;
    int requested{};
    int shape_requested{};
} fixture;
void reset() {
    sdk::reset();
    fixture = {};
}
Object& object(CudaDnnPooling native) {
    sdk::owner();
    for (auto& candidate : fixture.objects)
        if (static_cast<void*>(&candidate) == native) {
            expect(candidate.is_live, "pooling native identity must still be live");
            return candidate;
        }
    throw std::runtime_error("unknown native pooling descriptor");
}
template <typename Value>
void store_bits(Value* output, std::uint32_t bits) {
    static_assert(sizeof(Value) == sizeof(bits));
    std::memcpy(output, &bits, sizeof(bits));
}
CudaDnnDispatch dispatch() {
    auto result = sdk::dispatch();
    result.pooling_create = [](CudaDnnPooling* output) {
        sdk::owner();
        ++fixture.creates;
        if (fixture.has_null_success ||
            (fixture.create_status != CUDNN_STATUS_SUCCESS && !fixture.has_partial_create))
            return fixture.create_status;
        if (fixture.duplicate) {
            *output = static_cast<CudaDnnPooling>(fixture.duplicate);
            return fixture.create_status;
        }
        for (auto& candidate : fixture.objects)
            if (!candidate.is_live) {
                candidate = {};
                candidate.is_live = true;
                *output = reinterpret_cast<CudaDnnPooling>(&candidate);
                return fixture.create_status;
            }
        throw std::runtime_error("pooling native fixture capacity");
    };
    result.pooling_destroy = [](CudaDnnPooling native) {
        auto& candidate = object(native);
        expect(++candidate.destroys == 1, "uncertain native destruction is never retried");
        ++fixture.destroys;
        const auto status = fixture.fail_destroy_call == fixture.destroys
                                ? CUDNN_STATUS_INTERNAL_ERROR
                                : fixture.destroy_status;
        if (status == CUDNN_STATUS_SUCCESS) candidate.is_live = false;
        return status;
    };
    result.pooling_set_nd = [](CudaDnnPooling native, PoolingMode mode, cudnnNanPropagation_t nan,
                               int dimensions, const int* window, const int* padding,
                               const int* strides) {
        auto& candidate = object(native);
        ++fixture.sets;
        if (fixture.set_status != CUDNN_STATUS_SUCCESS) return fixture.set_status;
        if (dimensions > CUDNN_DIM_MAX - 2) return CUDNN_STATUS_NOT_SUPPORTED;
        if (dimensions <= 0) return CUDNN_STATUS_BAD_PARAM;
        for (int index = 0; index != dimensions; ++index)
            if (window[index] <= 0 || strides[index] <= 0)
                return CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
        auto& value = candidate.pooling;
        value.mode = static_cast<std::uint32_t>(mode);
        value.nan = static_cast<std::uint32_t>(nan);
        value.dimensions = dimensions;
        std::copy_n(window, dimensions, value.window.begin());
        std::copy_n(padding, dimensions, value.padding.begin());
        std::copy_n(strides, dimensions, value.strides.begin());
        return CUDNN_STATUS_SUCCESS;
    };
    result.pooling_get_nd = [](CudaDnnPooling native, int requested, PoolingMode* mode,
                               cudnnNanPropagation_t* nan, int* dimensions, int* window,
                               int* padding, int* strides) {
        const auto& value = object(native).pooling;
        ++fixture.gets;
        fixture.requested = requested;
        store_bits(mode, fixture.reported_mode.value_or(value.mode));
        store_bits(nan, fixture.reported_nan.value_or(value.nan));
        *dimensions = fixture.reported_rank.value_or(value.dimensions);
        const auto count = std::min(std::max(requested, 0), value.dimensions);
        std::copy_n(value.window.begin(), count, window);
        std::copy_n(value.padding.begin(), count, padding);
        std::copy_n(value.strides.begin(), count, strides);
        if (fixture.get_status != CUDNN_STATUS_SUCCESS) return fixture.get_status;
        if (!fixture.has_invalid_capacity_success) {
            if (requested < 0) return CUDNN_STATUS_BAD_PARAM;
            if (requested > CUDNN_DIM_MAX) return CUDNN_STATUS_NOT_SUPPORTED;
        }
        return CUDNN_STATUS_SUCCESS;
    };
    result.pooling_output_nd = [](CudaDnnPooling native, cudnnTensorDescriptor_t tensor,
                                  int dimensions, int* sizes) {
        const auto& pooling = object(native).pooling;
        const auto& input = sdk::object(tensor, Kind::kTensor).tensor;
        ++fixture.shapes;
        fixture.shape_requested = dimensions;
        std::fill_n(sizes, std::clamp(dimensions, 0, CUDNN_DIM_MAX), 47);
        if (fixture.shape_status != CUDNN_STATUS_SUCCESS) return fixture.shape_status;
        if (fixture.has_invalid_shape_success) return CUDNN_STATUS_SUCCESS;
        if (dimensions != input.dimensions || dimensions != pooling.dimensions + 2)
            return CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
        sizes[0] = input.sizes[0];
        sizes[1] = input.sizes[1];
        for (int index = 2; index != dimensions; ++index) {
            const auto spatial = static_cast<std::size_t>(index - 2);
            sizes[index] = 1 + (input.sizes[static_cast<std::size_t>(index)] +
                                2 * pooling.padding[spatial] - pooling.window[spatial]) /
                                   pooling.strides[spatial];
        }
        return CUDNN_STATUS_SUCCESS;
    };
    return result;
}
CudaDnnPoolingState value() {
    return {CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
            CUDNN_PROPAGATE_NAN,
            3,
            {2, 3, 4},
            {-1, 2, 0},
            {1, 2, 3}};
}
void metadata_and_original_errors() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto pooling = registry.create(Kind::kPooling).id;
    const auto tensor = registry.create(Kind::kTensor).id;
    expect(pooling && tensor && pooling != tensor, "descriptor kinds have distinct owned IDs");
    for (std::uint32_t mode = 0; mode != 4; ++mode) {
        auto pooling_value = value();
        pooling_value.mode = mode;
        pooling_value.nan = mode & 1;
        success(registry.set_pooling(pooling, pooling_value));
        for (int requested : {3, 7, 8}) {
            const auto result = registry.get_pooling(pooling, requested);
            success(result);
            expect(result.pooling.mode == mode && result.pooling.nan == pooling_value.nan &&
                       result.pooling.dimensions == 3 &&
                       result.pooling.window == pooling_value.window &&
                       result.pooling.padding == pooling_value.padding &&
                       result.pooling.strides == pooling_value.strides &&
                       fixture.requested == requested,
                   "every mode, negative padding and pooling-specific capacity are preserved");
        }
    }
    for (int requested : {0, 1}) {
        const auto result = registry.get_pooling(pooling, requested);
        success(result);
        expect(result.pooling.dimensions == 3 &&
                   result.pooling.window[static_cast<std::size_t>(requested)] == 0 &&
                   result.pooling.padding[static_cast<std::size_t>(requested)] == 0 &&
                   result.pooling.strides[static_cast<std::size_t>(requested)] == 0,
               "successful prefix queries publish no unwritten tail");
    }
    for (int requested :
         {std::numeric_limits<int>::min(), -1, 9, std::numeric_limits<int>::max()}) {
        original(registry.get_pooling(pooling, requested),
                 requested < 0 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_NOT_SUPPORTED);
        expect(fixture.requested == requested, "raw signed query capacity is not clipped");
    }
    success(registry.set_tensor(tensor, CudaDnnTensorSetter::kNdPacked,
                                {CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, {2, 3, 11, 13, 17}, {}}));
    const auto output = registry.pooling_output(pooling, tensor, 5);
    success(output);
    expect(output.tensor.dimensions == 5 &&
               output.tensor.sizes == std::array<int, CUDNN_DIM_MAX>{2, 3, 8, 8, 5},
           "same-owner output dimensions come from native pooling descriptors");
    original(registry.pooling_output(pooling, tensor, 4), CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH);
    for (int requested :
         {std::numeric_limits<int>::min(), -1, 9, std::numeric_limits<int>::max()}) {
        const auto invalid = registry.pooling_output(pooling, tensor, requested);
        original(invalid, CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH);
        expect(fixture.shape_requested == requested && !invalid.tensor.dimensions &&
                   !invalid.tensor.sizes[0],
               "raw signed output rank reaches the original error without partial publication");
    }
    fixture.get_status = CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH;
    const auto failed = registry.get_pooling(pooling, 3);
    original(failed, fixture.get_status);
    expect(!failed.pooling.dimensions && !failed.pooling.mode && !failed.pooling.window[0],
           "native getter failure publishes no touched scalar or array");
    fixture.shape_status = fixture.get_status;
    const auto failed_shape = registry.pooling_output(pooling, tensor, 5);
    original(failed_shape, fixture.shape_status);
    expect(!failed_shape.tensor.dimensions && !failed_shape.tensor.sizes[0],
           "native output-shape failure publishes no partial dimensions");
    success(registry.shutdown());
    expect(fixture.destroys == 1 && sdk::state.destroys == 1,
           "pooling and tensor metadata each destroy once");
    rejected(registry.get_pooling(pooling, 3), CUDA_ERROR_INVALID_HANDLE);
}
void validation_and_optional_callbacks() {
    reset();
    {
        CudaDnnRegistry old(sdk::dispatch());
        rejected(old.create(Kind::kPooling), CUDA_ERROR_NOT_SUPPORTED);
        success(old.shutdown());
    }
    auto callbacks = dispatch();
    callbacks.pooling_destroy = nullptr;
    {
        CudaDnnRegistry incomplete(callbacks);
        rejected(incomplete.create(Kind::kPooling), CUDA_ERROR_NOT_SUPPORTED);
        expect(!fixture.creates, "missing destruction prevents native ownership creation");
    }
    CudaDnnRegistry registry(dispatch(), 2);
    fixture.create_status = CUDNN_STATUS_ALLOC_FAILED;
    original(registry.create(Kind::kPooling), fixture.create_status);
    expect(!registry.snapshot().objects, "known empty creation failure retains no phantom owner");
    fixture.create_status = CUDNN_STATUS_SUCCESS;
    const auto pooling = registry.create(Kind::kPooling).id;
    const auto tensor = registry.create(Kind::kTensor).id;
    const auto creates = fixture.creates;
    rejected(registry.create(Kind::kPooling), CUDA_ERROR_OUT_OF_MEMORY);
    expect(fixture.creates == creates, "bounded capacity rejects before SDK creation");
    for (unsigned int mode = 0; mode != 4; ++mode) {
        auto invalid = value();
        if (mode == 0) invalid.mode = 4;
        if (mode == 1) invalid.nan = 2;
        if (mode == 2) invalid.dimensions = -1;
        if (mode == 3) invalid.dimensions = 9;
        rejected(registry.set_pooling(pooling, invalid), CUDA_ERROR_INVALID_VALUE);
    }
    rejected(registry.set_pooling(tensor, value()), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.get_pooling(999, 3), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.pooling_output(tensor, pooling, 5), CUDA_ERROR_INVALID_HANDLE);
    rejected(registry.pooling_output(pooling, 999, 5), CUDA_ERROR_INVALID_HANDLE);
    expect(!fixture.sets && !fixture.gets && !fixture.shapes,
           "unsafe raw enums, dimensions and identities reject before the SDK");
    for (int dimensions : {0, 7, 8}) {
        auto native_invalid = value();
        native_invalid.dimensions = dimensions;
        original(registry.set_pooling(pooling, native_invalid),
                 dimensions ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_BAD_PARAM);
    }
    auto native_invalid = value();
    native_invalid.window[0] = -1;
    original(registry.set_pooling(pooling, native_invalid), CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH);
    native_invalid = value();
    native_invalid.strides[0] = 0;
    original(registry.set_pooling(pooling, native_invalid), CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH);
    success(registry.set_pooling(pooling, value()));
    success(registry.destroy(pooling));
    rejected(registry.destroy(pooling), CUDA_ERROR_INVALID_HANDLE);
    success(registry.shutdown());

    for (unsigned int mode = 0; mode != 3; ++mode) {
        callbacks = dispatch();
        if (mode == 0) callbacks.pooling_set_nd = nullptr;
        if (mode == 1) callbacks.pooling_get_nd = nullptr;
        if (mode == 2) callbacks.pooling_output_nd = nullptr;
        CudaDnnRegistry incomplete(callbacks);
        const auto operation = incomplete.create(Kind::kPooling).id;
        const auto input = incomplete.create(Kind::kTensor).id;
        const auto result = mode == 0   ? incomplete.set_pooling(operation, value())
                            : mode == 1 ? incomplete.get_pooling(operation, 3)
                                        : incomplete.pooling_output(operation, input, 5);
        rejected(result, CUDA_ERROR_NOT_SUPPORTED);
        success(incomplete.shutdown());
    }
}
void borrowed_owner_lifecycle() {
    reset();
    CudaDnnRegistry registry(dispatch());
    const auto pooling = registry.create(Kind::kPooling).id;
    success(registry.set_pooling(pooling, value()));
    auto lease = registry.borrow(pooling, Kind::kPooling);
    expect(lease && !registry.borrow(pooling, Kind::kTensor), "typed pooling borrow is immutable");
    expect(registry.set_pooling(pooling, value()).disposition == CudaDnnDisposition::kDeferred &&
               registry.destroy(pooling).disposition == CudaDnnDisposition::kDeferred &&
               fixture.sets == 1 && !fixture.destroys,
           "leased pooling cannot mutate or destroy native state");
    success(registry.get_pooling(pooling, 3));
    const auto gets = fixture.gets;
    rejected(std::async(std::launch::async, [&] { return registry.get_pooling(pooling, 3); }).get(),
             CUDA_ERROR_UNKNOWN);
    const auto child = ::fork();
    expect(child >= 0, "fork pooling ownership control");
    if (!child) ::_exit(registry.get_pooling(pooling, 3).status == CUDA_ERROR_UNKNOWN ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               fixture.gets == gets,
           "foreign thread and fork never enter native descriptor SDK");
    registry.close();
    rejected(registry.set_pooling(pooling, value()), CUDA_ERROR_DEINITIALIZED);
    expect(!registry.borrow(pooling, Kind::kPooling) &&
               registry.shutdown().disposition == CudaDnnDisposition::kDeferred,
           "close blocks new pooling leases and waits for existing consumers");
    lease.reset();
    success(registry.shutdown());
    success(registry.shutdown());
    expect(fixture.destroys == 1, "quiescent pooling shutdown is exactly once");
}
void isolated_failures() {
    for (unsigned int mode = 0; mode != 17; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "fork isolated pooling uncertainty");
        if (!child) {
            try {
                reset();
                auto registry = std::make_unique<CudaDnnRegistry>(dispatch());
                const auto pooling = registry->create(Kind::kPooling).id;
                success(registry->set_pooling(pooling, value()));
                if (mode < 4) {
                    fixture.has_null_success = mode == 0;
                    fixture.has_partial_create = mode == 1;
                    if (mode == 1) fixture.create_status = CUDNN_STATUS_INTERNAL_ERROR;
                    if (mode >= 2) {
                        const auto id = mode == 2 ? pooling : registry->create(Kind::kTensor).id;
                        const auto view =
                            registry->borrow(id, mode == 2 ? Kind::kPooling : Kind::kTensor);
                        fixture.duplicate = view->native;
                    }
                    const auto result = registry->create(Kind::kPooling);
                    expect(result.disposition == CudaDnnDisposition::kRejected && !result.id &&
                               result.original_status == fixture.create_status,
                           "partial/null/cross-kind duplicate native creation cannot publish ID");
                } else if (mode == 4) {
                    fixture.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
                    original(registry->destroy(pooling), fixture.destroy_status);
                } else if (mode == 5) {
                    fixture.set_status = CUDNN_STATUS_EXECUTION_FAILED;
                    original(registry->set_pooling(pooling, value()), fixture.set_status);
                } else if (mode == 6) {
                    success(registry->create(Kind::kPooling));
                    fixture.fail_destroy_call = 2;
                    original(registry->shutdown(), CUDNN_STATUS_INTERNAL_ERROR);
                    expect(registry->snapshot().objects == 1 && fixture.destroys == 2,
                           "partial shutdown removes only successfully destroyed objects");
                } else {
                    if (mode == 7) fixture.reported_mode = 4;
                    if (mode == 8) fixture.reported_nan = 2;
                    if (mode == 9) fixture.reported_rank = -1;
                    if (mode == 10) fixture.reported_rank = 7;
                    if (mode == 11) fixture.reported_mode = UINT32_MAX;
                    if (mode == 12 || mode == 13) fixture.has_invalid_capacity_success = true;
                    if (mode == 14) fixture.reported_rank = 0;
                    CudaDnnResult result;
                    if (mode >= 15) {
                        fixture.has_invalid_shape_success = true;
                        const auto tensor = registry->create(Kind::kTensor).id;
                        result = registry->pooling_output(pooling, tensor, mode == 15 ? -1 : 9);
                    } else {
                        result = registry->get_pooling(pooling, mode == 12   ? -1
                                                                : mode == 13 ? 9
                                                                             : 3);
                    }
                    expect(result.disposition == CudaDnnDisposition::kRejected &&
                               result.status == CUDA_ERROR_UNKNOWN &&
                               result.original_status == CUDNN_STATUS_SUCCESS &&
                               !result.pooling.dimensions && !result.pooling.window[0] &&
                               !result.tensor.dimensions && !result.tensor.sizes[0],
                           "malformed successful getter quarantines without partial publication");
                }
                expect(registry->snapshot().is_quarantined, "pooling uncertainty is explicit");
                const auto destroys = fixture.destroys, gets = fixture.gets, sets = fixture.sets;
                rejected(registry->get_pooling(pooling, 3), CUDA_ERROR_UNKNOWN);
                rejected(registry->set_pooling(pooling, value()), CUDA_ERROR_UNKNOWN);
                rejected(registry->shutdown(), CUDA_ERROR_UNKNOWN);
                registry.reset();
                expect(fixture.destroys == destroys && fixture.gets == gets && fixture.sets == sets,
                       "quarantine never retries uncertain SDK mutations or cleanup");
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "pooling quarantine mode " << mode << ": " << error.what() << '\n';
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated pooling lifecycle and malformed-output controls pass");
    }
}
}  // namespace
int main() {
    try {
        metadata_and_original_errors();
        validation_and_optional_callbacks();
        borrowed_owner_lifecycle();
        isolated_failures();
        std::cout << "cuDNN pooling registry ownership checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
