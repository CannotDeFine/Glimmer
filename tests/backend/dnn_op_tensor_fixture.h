#pragma once

#include "dnn_convolution_fixture.h"

#include <chrono>
// Independent CPU ABI/metadata oracle. It never invokes the vendor library and
// does not qualify GPU execution or the vendor's complete folding algorithm.
namespace glimmer::tests::dnn_op_tensor {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace driver = glimmer::tests::dnn_convolution;
using Kind = CudaDnnObjectKind;
struct Object {
    bool alive{};
    Kind kind{};
    CudaDnnOpTensorState operation{CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN};
    CudaDnnTensorTransformState transform{};
    unsigned int destroy_attempts{};
};
struct State {
    std::array<Object, 16> objects{};
    unsigned int creates{}, destroys{}, sets{}, gets{}, initializes{};
    unsigned int query_masks{}, array_masks{};
    CudaDnnTensorTransformState last_transform{};
    CudaDnnTensorTransformState wide_transform{};
    void* last_source{};
    void* last_destination{};
    bool last_has_size{};
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, destroy_status{CUDNN_STATUS_SUCCESS};
    bool partial_create{}, null_create{}, duplicate_create{};
};
struct Control {
    std::atomic<int> get_status{CUDNN_STATUS_SUCCESS}, set_status{CUDNN_STATUS_SUCCESS},
        init_status{CUDNN_STATUS_SUCCESS};
    std::atomic<std::uint32_t> invalid_output_mask{};
    std::atomic<std::size_t> reported_bytes{};
    std::atomic<unsigned int> callbacks{}, device_uuid_calls{}, driver_version_calls{};
    std::atomic<bool> pause_init{}, init_entered{}, release_init{true};
};
inline State state;
inline Control control;
inline void reset() {
    driver::reset();
    state = {};
    control.get_status.store(CUDNN_STATUS_SUCCESS);
    control.set_status.store(CUDNN_STATUS_SUCCESS);
    control.init_status.store(CUDNN_STATUS_SUCCESS);
    control.invalid_output_mask.store(0);
    control.reported_bytes.store(0);
    control.callbacks.store(0);
    control.device_uuid_calls.store(0);
    control.driver_version_calls.store(0);
    control.pause_init.store(false);
    control.init_entered.store(false);
    control.release_init.store(true);
}
inline Object& object(void* native, Kind kind) {
    base::owner();
    for (auto& entry : state.objects)
        if (&entry == native) {
            base::require(entry.alive && entry.kind == kind);
            return entry;
        }
    throw std::runtime_error("unknown OpTensor/Transform native object");
}
inline Object* create(Kind kind) {
    base::owner();
    ++state.creates;
    if (state.null_create || (state.create_status != CUDNN_STATUS_SUCCESS && !state.partial_create))
        return nullptr;
    if (state.duplicate_create)
        for (auto& entry : state.objects)
            if (entry.alive && entry.kind == kind) return &entry;
    for (auto& entry : state.objects)
        if (!entry.alive) {
            entry = {};
            entry.alive = true;
            entry.kind = kind;
            return &entry;
        }
    throw std::runtime_error("OpTensor/Transform fixture capacity");
}
inline cudnnStatus_t destroy(void* native, Kind kind) {
    auto& entry = object(native, kind);
    ++state.destroys;
    base::require(++entry.destroy_attempts == 1);
    if (state.destroy_status == CUDNN_STATUS_SUCCESS) entry.alive = false;
    return state.destroy_status;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = base::dispatch();
    sdk.op_tensor_create = [](CudaDnnOpTensor* output) {
        *output = reinterpret_cast<CudaDnnOpTensor>(create(Kind::kOpTensor));
        return state.create_status;
    };
    sdk.op_tensor_destroy = [](CudaDnnOpTensor native) { return destroy(native, Kind::kOpTensor); };
    sdk.tensor_transform_create = [](CudaDnnTensorTransform* output) {
        *output = reinterpret_cast<CudaDnnTensorTransform>(create(Kind::kTensorTransform));
        return state.create_status;
    };
    sdk.tensor_transform_destroy = [](CudaDnnTensorTransform native) {
        return destroy(native, Kind::kTensorTransform);
    };
    sdk.op_tensor_set = [](CudaDnnOpTensor native, cudnnOpTensorOp_t operation,
                           cudnnDataType_t type, cudnnNanPropagation_t nan) {
        auto& entry = object(native, Kind::kOpTensor);
        ++state.sets;
        ++control.callbacks;
        if (control.set_status.load() != CUDNN_STATUS_SUCCESS)
            return static_cast<cudnnStatus_t>(control.set_status.load());
        if (operation > CUDNN_OP_TENSOR_NOT || type > CUDNN_DATA_DOUBLE)
            return CUDNN_STATUS_BAD_PARAM;
        entry.operation = {static_cast<std::uint32_t>(operation), static_cast<std::uint32_t>(type),
                           static_cast<std::uint32_t>(nan)};
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.op_tensor_get = [](CudaDnnOpTensor native, cudnnOpTensorOp_t* operation,
                           cudnnDataType_t* type, cudnnNanPropagation_t* nan) {
        const auto& entry = object(native, Kind::kOpTensor);
        ++state.gets;
        const unsigned int mask = (operation ? 1U : 0U) | (type ? 2U : 0U) | (nan ? 4U : 0U);
        state.query_masks |= 1U << mask;
        const auto malformed = control.invalid_output_mask.load();
        const std::uint32_t invalid = 0xffffffffU;
        if (operation)
            std::memcpy(operation, malformed & 1U ? &invalid : &entry.operation.operation, 4);
        if (type) std::memcpy(type, malformed & 2U ? &invalid : &entry.operation.type, 4);
        if (nan) std::memcpy(nan, malformed & 4U ? &invalid : &entry.operation.nan, 4);
        return static_cast<cudnnStatus_t>(control.get_status.load());
    };
    sdk.tensor_transform_set = [](CudaDnnTensorTransform native, std::uint32_t dimensions,
                                  cudnnTensorFormat_t format, const std::int32_t* before,
                                  const std::int32_t* after, const std::uint32_t* fold,
                                  cudnnFoldingDirection_t direction) {
        auto& entry = object(native, Kind::kTensorTransform);
        ++state.sets;
        ++control.callbacks;
        base::require(dimensions <= 8);
        CudaDnnTensorTransformState input{dimensions,
                                          static_cast<std::uint32_t>(format),
                                          static_cast<std::uint32_t>(direction),
                                          (before ? 1U : 0U) | (after ? 2U : 0U) | (fold ? 4U : 0U),
                                          {},
                                          {},
                                          {}};
        if (before) std::copy_n(before, dimensions, input.before.begin());
        if (after) std::copy_n(after, dimensions, input.after.begin());
        if (fold && dimensions > 2) std::copy_n(fold, dimensions - 2, input.fold.begin());
        state.last_transform = input;
        if (dimensions == 8 && input.presence == 7) state.wide_transform = input;
        state.array_masks |= 1U << input.presence;
        if (control.set_status.load() != CUDNN_STATUS_SUCCESS)
            return static_cast<cudnnStatus_t>(control.set_status.load());
        if (dimensions <= 2 || format > CUDNN_TENSOR_NHWC) return CUDNN_STATUS_BAD_PARAM;
        entry.transform = input;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.init_transform_dest = [](CudaDnnTensorTransform native, cudnnTensorDescriptor_t source,
                                 cudnnTensorDescriptor_t destination, std::size_t* bytes) {
        const auto transform = object(native, Kind::kTensorTransform).transform;
        // Snapshot before mutating to make the exact native alias observable.
        const auto input = base::object(source, Kind::kTensor).tensor;
        auto& output = base::object(destination, Kind::kTensor);
        ++state.initializes;
        ++control.callbacks;
        if (control.pause_init.exchange(false)) {
            control.init_entered.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (!control.release_init.load()) {
                base::require(std::chrono::steady_clock::now() < deadline);
                std::this_thread::yield();
            }
        }
        state.last_source = source;
        state.last_destination = destination;
        state.last_has_size = bytes;
        output.tensor = input;
        output.configured = true;
        if (bytes) *bytes = 0xbadbad;
        const auto status = static_cast<cudnnStatus_t>(control.init_status.load());
        if (status != CUDNN_STATUS_SUCCESS) {
            output.tensor.sizes[0] = 777;  // Deliberate partial mutation on error.
            return status;
        }
        base::require(input.dimensions > 2 && input.dimensions <= 8 &&
                      transform.dimensions == static_cast<std::uint32_t>(input.dimensions));
        std::size_t elements = 1;
        for (int index = 0; index != input.dimensions; ++index) {
            const auto i = static_cast<std::size_t>(index);
            output.tensor.sizes[i] += transform.before[i] + transform.after[i];
            base::require(output.tensor.sizes[i] > 0);
            elements *= static_cast<unsigned int>(output.tensor.sizes[i]);
        }
        output.tensor.format = transform.format;
        int stride = 1;
        if (transform.format == CUDNN_TENSOR_NHWC) {
            output.tensor.strides[1] = stride;
            stride *= output.tensor.sizes[1];
            for (int index = input.dimensions - 1; index >= 2; --index) {
                const auto i = static_cast<std::size_t>(index);
                output.tensor.strides[i] = stride;
                stride *= output.tensor.sizes[i];
            }
            output.tensor.strides[0] = stride;
        } else {
            for (int index = input.dimensions - 1; index >= 0; --index) {
                const auto i = static_cast<std::size_t>(index);
                output.tensor.strides[i] = stride;
                stride *= output.tensor.sizes[i];
            }
        }
        output.bytes = elements * (input.type == CUDNN_DATA_DOUBLE ? 8U : 4U);
        if (bytes)
            *bytes = control.reported_bytes.load() ? control.reported_bytes.load() : output.bytes;
        return CUDNN_STATUS_SUCCESS;
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = driver::session_dispatch();
    // Profile construction requires these capabilities even for a DNM1-only
    // client. Their counters prove that metadata never queries a GPU identity.
    sdk.device_uuid = [](CUuuid* output, CUdevice) {
        base::owner();
        ++control.device_uuid_calls;
        *output = {};
        output->bytes[0] = 19;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* output) {
        base::owner();
        ++control.driver_version_calls;
        *output = 12060;
        return CUDA_SUCCESS;
    };
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_op_tensor
