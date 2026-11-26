#include "internal/cuda_lt_registry.h"

#include <array>
#include <algorithm>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Object {
    bool alive{};
    CudaLtObjectKind kind{};
    std::array<std::array<std::byte, 16>, 32> attributes{};
};
struct State {
    std::array<Object, 32> objects{};
    unsigned int creates{}, destroys{}, sets{}, gets{}, versions{};
    cublasStatus_t create_status{CUBLAS_STATUS_SUCCESS}, destroy_status{CUBLAS_STATUS_SUCCESS},
        set_status{CUBLAS_STATUS_SUCCESS}, get_status{CUBLAS_STATUS_SUCCESS};
    bool partial_create{}, null_create{}, wrong_size{}, wrong_pointer{};
    unsigned int capabilities{};
    std::size_t sdk_version{120604};
    bool capability_error{}, capability_bad_count{};
} state;
template <typename Value>
void store(Object& object, unsigned int attribute, Value value) {
    std::memcpy(object.attributes[attribute].data(), &value, sizeof(value));
}
Object* native_create(CudaLtObjectKind kind) {
    ++state.creates;
    if (state.null_create ||
        (state.create_status != CUBLAS_STATUS_SUCCESS && !state.partial_create))
        return nullptr;
    for (auto& object : state.objects)
        if (!object.alive) {
            object = {};
            object.alive = true;
            object.kind = kind;
            return &object;
        }
    throw std::runtime_error("fake native registry capacity");
}
cublasStatus_t native_destroy(void* pointer) {
    ++state.destroys;
    auto& object = *static_cast<Object*>(pointer);
    expect(object.alive, "native metadata is destroyed at most once");
    if (state.destroy_status == CUBLAS_STATUS_SUCCESS) object.alive = false;
    return state.destroy_status;
}
cublasStatus_t native_set(void* pointer, unsigned int attribute, const void* bytes,
                          std::size_t size) {
    ++state.sets;
    auto& object = *static_cast<Object*>(pointer);
    expect(object.alive && attribute < object.attributes.size(),
           "only live supported native metadata");
    expect(reinterpret_cast<std::uintptr_t>(bytes) % alignof(std::uint64_t) == 0,
           "original SDK receives aligned immutable attribute bytes");
    if (state.set_status == CUBLAS_STATUS_SUCCESS)
        std::memcpy(object.attributes[attribute].data(), bytes, size);
    return state.set_status;
}
cublasStatus_t native_get(void* pointer, unsigned int attribute, void* bytes, std::size_t size,
                          std::size_t* written) {
    ++state.gets;
    auto& object = *static_cast<Object*>(pointer);
    expect(object.alive, "query uses a live native object");
    const auto description = cuda_lt_attribute(object.kind, attribute);
    expect(description.has_value(), "query classified before native SDK issue");
    if (state.get_status != CUBLAS_STATUS_SUCCESS) return state.get_status;
    if (size && size != description->bytes) return CUBLAS_STATUS_INVALID_VALUE;
    *written = description->bytes + (state.wrong_size ? 1 : 0);
    if (size) {
        std::memcpy(bytes, object.attributes[attribute].data(), description->bytes);
        if (description->is_device_pointer && state.wrong_pointer) {
            const std::uint64_t invalid = 0xdeadbeef;
            std::memcpy(bytes, &invalid, sizeof(invalid));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}
CudaLtDispatch dispatch() {
    CudaLtDispatch result;
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): original injected SDK ABIs.
    result.layout_create = [](cublasLtMatrixLayout_t* output, cudaDataType type, std::uint64_t rows,
                              std::uint64_t columns, std::int64_t leading) {
        auto* object = native_create(CudaLtObjectKind::kLayout);
        *output = reinterpret_cast<cublasLtMatrixLayout_t>(object);
        if (object) {
            store(*object, CUBLASLT_MATRIX_LAYOUT_TYPE, type);
            store(*object, CUBLASLT_MATRIX_LAYOUT_ROWS, rows);
            store(*object, CUBLASLT_MATRIX_LAYOUT_COLS, columns);
            store(*object, CUBLASLT_MATRIX_LAYOUT_LD, leading);
        }
        return state.create_status;
    };
    result.layout_destroy = [](cublasLtMatrixLayout_t object) { return native_destroy(object); };
    result.layout_set = [](cublasLtMatrixLayout_t object, cublasLtMatrixLayoutAttribute_t attribute,
                           const void* value,
                           std::size_t size) { return native_set(object, attribute, value, size); };
    result.layout_get = [](cublasLtMatrixLayout_t object, cublasLtMatrixLayoutAttribute_t attribute,
                           void* value, std::size_t size, std::size_t* written) {
        return native_get(object, attribute, value, size, written);
    };
    result.operation_create = [](cublasLtMatmulDesc_t* output, cublasComputeType_t compute,
                                 cudaDataType scale) {
        auto* object = native_create(CudaLtObjectKind::kOperation);
        *output = reinterpret_cast<cublasLtMatmulDesc_t>(object);
        if (object) {
            store(*object, CUBLASLT_MATMUL_DESC_COMPUTE_TYPE, compute);
            store(*object, CUBLASLT_MATMUL_DESC_SCALE_TYPE, scale);
        }
        return state.create_status;
    };
    result.operation_destroy = [](cublasLtMatmulDesc_t object) { return native_destroy(object); };
    result.operation_set = [](cublasLtMatmulDesc_t object, cublasLtMatmulDescAttributes_t attribute,
                              const void* value, std::size_t size) {
        return native_set(object, attribute, value, size);
    };
    result.operation_get = [](cublasLtMatmulDesc_t object, cublasLtMatmulDescAttributes_t attribute,
                              void* value, std::size_t size, std::size_t* written) {
        return native_get(object, attribute, value, size, written);
    };
    result.preference_create = [](cublasLtMatmulPreference_t* output) {
        *output = reinterpret_cast<cublasLtMatmulPreference_t>(
            native_create(CudaLtObjectKind::kPreference));
        return state.create_status;
    };
    result.preference_destroy = [](cublasLtMatmulPreference_t object) {
        return native_destroy(object);
    };
    result.preference_set = [](cublasLtMatmulPreference_t object,
                               cublasLtMatmulPreferenceAttributes_t attribute, const void* value,
                               std::size_t size) {
        return native_set(object, attribute, value, size);
    };
    result.preference_get = [](cublasLtMatmulPreference_t object,
                               cublasLtMatmulPreferenceAttributes_t attribute, void* value,
                               std::size_t size, std::size_t* written) {
        return native_get(object, attribute, value, size, written);
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    result.version = []() -> std::size_t {
        ++state.versions;
        return state.sdk_version;
    };
    result.runtime_version = []() -> std::size_t {
        ++state.versions;
        return 12060;
    };
    result.capability_get = [](const cublasLtMatmulAlgo_t* algorithm,
                               cublasLtMatmulAlgoCapAttributes_t attribute, void* bytes,
                               std::size_t count, std::size_t* written) {
        ++state.capabilities;
        expect(algorithm && algorithm->data[0] == 77,
               "only exact remembered original value reaches SDK");
        if (state.capability_error) {
            if (bytes && count) *static_cast<std::byte*>(bytes) = std::byte{0xee};
            if (written) *written = 99;
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        const std::array<std::uint32_t, 3> values{3, 7, 11};
        const auto needed =
            attribute == CUBLASLT_ALGO_CAP_TILE_IDS ? sizeof(values) : sizeof(values[0]);
        if (!count && written) {
            *written = needed;
            return CUBLAS_STATUS_SUCCESS;
        }
        if (!bytes || count < needed) return CUBLAS_STATUS_INVALID_VALUE;
        std::memcpy(bytes, values.data(), needed);
        if (written) *written = state.capability_bad_count ? count + 1 : needed;
        return CUBLAS_STATUS_SUCCESS;
    };
    return result;
}
void success(const CudaLtResult& result) {
    expect(result.disposition == CudaLtDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUBLAS_STATUS_SUCCESS,
           "actual original Lt result, not fabricated SDK success");
}
template <typename Value>
auto bytes(const Value& value) {
    return std::as_bytes(std::span{&value, std::size_t{1}});
}
void resources_and_native_queries() {
    state = {};
    CudaLtRegistry registry(dispatch());
    expect(registry.version(false).value == 120604 && registry.version(true).value == 12060 &&
               !registry.version(false).original_status,
           "size_t version returns never claim a cublasStatus callback");
    const auto layout = registry.create({.rows = 17, .columns = 19, .leading = 32});
    const auto operation = registry.create({.kind = CudaLtObjectKind::kOperation});
    const auto preference = registry.create({.kind = CudaLtObjectKind::kPreference});
    success(layout);
    success(operation);
    success(preference);
    expect(registry.snapshot().objects == 3 && state.creates == 3,
           "three actual metadata kinds need no CUDA context");
    std::int8_t fast{1};
    success(registry.set_attribute({operation.id, CUBLASLT_MATMUL_DESC_FAST_ACCUM}, bytes(fast)));
    const auto fast_query =
        registry.get_attribute({operation.id, CUBLASLT_MATMUL_DESC_FAST_ACCUM}, 1);
    success(fast_query);
    expect(fast_query.value == 1 && fast_query.data.size() == 1 &&
               fast_query.data.front() == std::byte{1},
           "original FAST_ACCUM storage is one byte, not a 32-bit enum");
    auto size = registry.get_attribute({layout.id, CUBLASLT_MATRIX_LAYOUT_ROWS}, 0);
    success(size);
    expect(size.value == 8 && size.data.empty(), "native zero-buffer size query");
    auto rows = registry.get_attribute({layout.id, CUBLASLT_MATRIX_LAYOUT_ROWS}, 8);
    success(rows);
    std::uint64_t observed{};
    std::memcpy(&observed, rows.data.data(), sizeof(observed));
    expect(observed == 17, "actual original native layout output");
    auto invalid = registry.get_attribute({layout.id, CUBLASLT_MATRIX_LAYOUT_ROWS}, 1);
    expect(invalid.original_status == CUBLAS_STATUS_INVALID_VALUE && invalid.data.empty(),
           "original size error never copies partial output bytes");
    std::uint32_t order = CUBLASLT_ORDER_ROW;
    success(registry.set_attribute({layout.id, CUBLASLT_MATRIX_LAYOUT_ORDER}, bytes(order)));
    order = 99;
    auto queried = registry.get_attribute({layout.id, CUBLASLT_MATRIX_LAYOUT_ORDER}, 4);
    success(queried);
    std::memcpy(&order, queried.data.data(), sizeof(order));
    expect(order == CUBLASLT_ORDER_ROW, "native descriptor owns bytes after caller reuse");
    state.set_status = CUBLAS_STATUS_NOT_SUPPORTED;
    expect(registry.set_attribute({layout.id, CUBLASLT_MATRIX_LAYOUT_ORDER}, bytes(order))
                       .original_status == CUBLAS_STATUS_NOT_SUPPORTED &&
               !registry.snapshot().is_quarantined,
           "known original rejection retains a usable descriptor");
    state.set_status = CUBLAS_STATUS_SUCCESS;
    success(registry.destroy(layout.id));
    success(registry.destroy(operation.id));
    success(registry.destroy(preference.id));
    expect(state.destroys == 3 && registry.shutdown().disposition == CudaLtDisposition::kComplete,
           "explicit metadata shutdown destroys each native object exactly once");
}
void bounds_and_nonrecycled_identities() {
    state = {};
    CudaLtRegistry registry(dispatch(), 2);
    state.create_status = CUBLAS_STATUS_ALLOC_FAILED;
    expect(registry.create({}).original_status == CUBLAS_STATUS_ALLOC_FAILED &&
               registry.snapshot().objects == 0,
           "known-empty original rejection does not fund or retain a published resource");
    state.create_status = CUBLAS_STATUS_SUCCESS;
    const auto first = registry.create({}), second = registry.create({});
    success(first);
    success(second);
    const auto creates = state.creates;
    expect(registry.create({}).status == CUDA_ERROR_OUT_OF_MEMORY && state.creates == creates,
           "metadata bound rejects before native issue");
    success(registry.destroy(first.id));
    const auto third = registry.create({});
    success(third);
    expect(third.id > second.id && first.id < second.id &&
               registry.get_attribute({first.id, 0}, 4).status == CUDA_ERROR_INVALID_HANDLE,
           "retired native addresses may recycle, but public resource identities never do");
}
void immutable_borrows_and_pointer_authority() {
    state = {};
    CudaLtRegistry registry(dispatch());
    const auto operation = registry.create({.kind = CudaLtObjectKind::kOperation});
    success(operation);
    std::array<std::byte, 8> neutral{};
    CudaLtDeviceReference reference{41, 9, 8, 0x12000};
    success(registry.set_attribute({operation.id, CUBLASLT_MATMUL_DESC_BIAS_POINTER}, neutral,
                                   reference));
    auto view = registry.borrow(operation.id, CudaLtObjectKind::kOperation);
    expect(view && view->bias && view->bias->allocation == 9 &&
               view->bias->address == reference.address && registry.snapshot().borrowed == 1,
           "immutable native borrow retains authenticated bias identity");
    expect(!registry.borrow(operation.id, CudaLtObjectKind::kLayout),
           "cross-kind object guesses reject");
    const auto native_sets = state.sets;
    expect(registry.set_attribute({operation.id, CUBLASLT_MATMUL_DESC_BIAS_POINTER}, neutral)
                       .disposition == CudaLtDisposition::kDeferred &&
               registry.destroy(operation.id).disposition == CudaLtDisposition::kDeferred &&
               state.sets == native_sets && state.destroys == 0,
           "host metadata cannot mutate or destroy a descriptor retained by actual GPU work");
    success(registry.get_attribute({operation.id, CUBLASLT_MATMUL_DESC_BIAS_POINTER}, 8));
    view.reset();
    neutral[0] = std::byte{1};
    expect(
        registry.set_attribute({operation.id, CUBLASLT_MATMUL_DESC_BIAS_POINTER}, neutral).status ==
                CUDA_ERROR_INVALID_VALUE &&
            state.sets == native_sets,
        "unclassified caller pointer bytes cannot reach the SDK");
    neutral[0] = std::byte{};
    expect(registry.set_attribute({operation.id, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER},
                                  neutral, reference)
                   .status == CUDA_ERROR_NOT_SUPPORTED,
           "unclassified auxiliary-pointer contracts fail closed");
    view = registry.borrow(operation.id, CudaLtObjectKind::kOperation);
    registry.close();
    expect(registry.shutdown().disposition == CudaLtDisposition::kDeferred &&
               !registry.snapshot().is_closed && state.destroys == 0 &&
               !registry.borrow(operation.id, CudaLtObjectKind::kOperation),
           "close seals new borrows but never invents native retirement");
    view.reset();
    expect(registry.shutdown().disposition == CudaLtDisposition::kComplete &&
               registry.snapshot().is_closed && state.destroys == 1,
           "release of the actual last command borrow permits once-only owner cleanup");
}
template <typename Function>
void isolated(Function&& function) {
    const auto child = ::fork();
    expect(child >= 0, "isolated native ownership fixture");
    if (!child) {
        try {
            function();
            ::_exit(0);
        } catch (...) {
            ::_exit(1);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated uncertain ownership control must pass, not claim leak-free recovery");
}
void algorithm_capability_values() {
    state = {};
    CudaLtRegistry registry(dispatch(), 2);
    std::array<cublasLtMatmulHeuristicResult_t, 3> algorithms{};
    for (std::size_t index = 0; index < algorithms.size(); ++index) {
        algorithms[index].state = CUBLAS_STATUS_SUCCESS;
        algorithms[index].algo.data[0] = 77 + index;
    }
    const auto& algorithm = algorithms[0].algo;
    std::array<std::byte, 16> initial;
    initial.fill(std::byte{0xa5});
    const auto query = [&](std::uint32_t attribute, std::size_t count = 16,
                           std::uint32_t presence = 3) {
        return registry.algorithm_capability(algorithm, attribute, count, presence,
                                             (presence & 1U)
                                                 ? std::span<const std::byte>{initial}.first(count)
                                                 : std::span<const std::byte>{},
                                             0);
    };
    expect(registry.remember_algorithms(algorithms, 120604) == CUDA_ERROR_OUT_OF_MEMORY &&
               query(CUBLASLT_ALGO_CAP_POINTER_MODE_MASK).status == CUDA_ERROR_NOT_SUPPORTED &&
               state.capabilities == 0,
           "cache exhaustion publishes no partial algorithm provenance or SDK call");
    expect(registry.remember_algorithms(std::span{algorithms}.first(1), 120604) == CUDA_SUCCESS &&
               registry.remember_algorithms(std::span{algorithms}.first(1), 120604) == CUDA_SUCCESS,
           "complete original value cache is bounded and duplicate idempotent");
    const auto scalar = query(CUBLASLT_ALGO_CAP_POINTER_MODE_MASK);
    success(scalar);
    std::uint32_t mask{};
    std::memcpy(&mask, scalar.data.data(), sizeof(mask));
    expect(mask == 3 && scalar.value == 4 && scalar.data.size() == initial.size() &&
               std::ranges::all_of(std::span{scalar.data}.subspan(4),
                                   [](auto value) { return value == std::byte{0xa5}; }),
           "actual SDK scalar and untouched complete output tail are preserved");
    const auto array = query(CUBLASLT_ALGO_CAP_TILE_IDS);
    success(array);
    expect(array.value == 12 && array.data.size() == 16 && array.data[12] == std::byte{0xa5},
           "bounded array capabilities retain all output bytes");
    const auto count = query(CUBLASLT_ALGO_CAP_TILE_IDS, 0, 2);
    success(count);
    expect(count.value == 12 && count.data.empty(), "zero-size count query needs no output buffer");
    const auto absent = query(CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, 4, 2);
    expect(absent.original_status == CUBLAS_STATUS_INVALID_VALUE && absent.data.empty(),
           "nonzero-size NULL buffer remains an original SDK error");
    const auto before = state.capabilities;
    auto forged = algorithm;
    forged.data[1] = 1;
    expect(registry.algorithm_capability(forged, 11, 16, 3, initial, 0).status ==
                   CUDA_ERROR_NOT_SUPPORTED &&
               query(9).status == CUDA_ERROR_NOT_SUPPORTED &&
               registry.algorithm_capability(algorithm, 11, 4097, 2, {}, 0).status ==
                   CUDA_ERROR_INVALID_VALUE &&
               state.capabilities == before,
           "unknown algorithms/attribute holes and oversized frames reject before SDK entry");
    state.capability_error = true;
    const auto error = query(CUBLASLT_ALGO_CAP_POINTER_MODE_MASK);
    expect(
        error.status == CUDA_SUCCESS && error.original_status == CUBLAS_STATUS_INVALID_VALUE &&
            error.value == 99 && error.data[0] == std::byte{0xee} &&
            error.data[1] == std::byte{0xa5},
        "exact native error-side buffer/count effects are returned, not fabricated clean failure");
    state.capability_error = false;
    const auto calls = state.capabilities;
    expect(std::async(std::launch::async, [&] { return query(11).status; }).get() ==
                   CUDA_ERROR_UNKNOWN &&
               state.capabilities == calls,
           "foreign thread cannot consume recorded provenance");
    expect(registry.shutdown().status == CUDA_SUCCESS,
           "pure algorithm provenance does not delay metadata shutdown");
    success(query(CUBLASLT_ALGO_CAP_POINTER_MODE_MASK));
    expect(registry.snapshot().objects == 0 && state.creates == 0,
           "pure algorithm values outlive all descriptor/GPU handles without creating resources");
    isolated([] {
        state = {};
        CudaLtRegistry owned(dispatch());
        cublasLtMatmulHeuristicResult_t item{};
        item.algo.data[0] = 77;
        expect(owned.remember_algorithms(std::span{&item, 1}, 120604) == CUDA_SUCCESS,
               "record valid algorithm");
        state.capability_bad_count = true;
        const std::array<std::byte, 4> output{};
        const auto invalid = owned.algorithm_capability(item.algo, 11, 4, 3, output, 0);
        expect(invalid.status == CUDA_ERROR_UNKNOWN &&
                   invalid.original_status == CUBLAS_STATUS_SUCCESS && invalid.data.empty() &&
                   owned.snapshot().is_quarantined,
               "contradictory native success quarantines without publishing or retrying output");
    });
}
void owner_guards_and_unknown_ownership() {
    state = {};
    CudaLtRegistry registry(dispatch());
    const auto rejected =
        std::async(std::launch::async, [&] { return registry.create({}).status; }).get();
    expect(rejected == CUDA_ERROR_UNKNOWN && state.creates == 0,
           "foreign native thread never calls the SDK");
    isolated([&] {
        expect(registry.version(false).status == CUDA_ERROR_UNKNOWN,
               "fork cannot use inherited registry");
    });
    for (unsigned int mode = 0; mode < 5; ++mode)
        isolated([mode] {
            state = {};
            CudaLtRegistry owned(dispatch());
            if (mode < 2) {
                state.partial_create = mode == 0;
                state.null_create = mode == 1;
                state.create_status =
                    mode == 0 ? CUBLAS_STATUS_ALLOC_FAILED : CUBLAS_STATUS_SUCCESS;
                const auto failed = owned.create({});
                expect(failed.status == CUDA_ERROR_UNKNOWN && owned.snapshot().is_quarantined &&
                           owned.snapshot().objects == 1,
                       "partial native creation and success-without-handle retain uncertain "
                       "ownership");
            } else {
                const auto object = owned.create({.kind = CudaLtObjectKind::kOperation});
                success(object);
                if (mode == 2) {
                    state.destroy_status = CUBLAS_STATUS_EXECUTION_FAILED;
                    expect(
                        owned.destroy(object.id).original_status == CUBLAS_STATUS_EXECUTION_FAILED,
                        "original destroy error remains original");
                } else {
                    state.wrong_size = mode == 3;
                    state.wrong_pointer = mode == 4;
                    const auto result =
                        owned.get_attribute({object.id, CUBLASLT_MATMUL_DESC_BIAS_POINTER}, 8);
                    expect(result.status == CUDA_ERROR_UNKNOWN &&
                               result.original_status == CUBLAS_STATUS_SUCCESS &&
                               result.data.empty(),
                           "contradictory successful SDK output cannot create pointer or size "
                           "authority");
                }
                expect(owned.snapshot().is_quarantined,
                       "unknown SDK state quarantines further use");
            }
            const auto destroys = state.destroys, creates = state.creates;
            expect(owned.create({}).status == CUDA_ERROR_UNKNOWN &&
                       owned.destroy(1).status == CUDA_ERROR_UNKNOWN &&
                       owned.shutdown().status == CUDA_ERROR_UNKNOWN && state.creates == creates &&
                       state.destroys == destroys,
                   "unknown native state never retries, replays or releases ownership");
        });
}
}  // namespace
int main() {
    try {
        resources_and_native_queries();
        bounds_and_nonrecycled_identities();
        immutable_borrows_and_pointer_authority();
        algorithm_capability_values();
        owner_guards_and_unknown_ownership();
        std::cout << "Owned Lt metadata verified: native_status=true; immutable_borrows=true; "
                     "cuda_context_required=false; gpu_qualified=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
