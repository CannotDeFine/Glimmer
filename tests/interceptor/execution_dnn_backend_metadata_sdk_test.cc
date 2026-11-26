#include <cudnn.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

// Ordinary SDK metadata input: no Glimmer API, CUDA context, allocation or
// execution. Run explicitly; this is neither a GPU nor an owned-route test.
namespace {
unsigned int calls{};
bool has_cleanup_failure{};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(cudnnStatus_t result, const char* operation) {
    ++calls;
    if (result != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(result)));
}
class Descriptor final {
   public:
    explicit Descriptor(cudnnBackendDescriptorType_t type) {
        checked(cudnnBackendCreateDescriptor(type, &value_), "create backend descriptor");
        expect(value_ != nullptr, "successful native creation returns a descriptor");
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    ~Descriptor() {
        if (value_) {
            ++calls;
            if (cudnnBackendDestroyDescriptor(std::exchange(value_, nullptr)) !=
                CUDNN_STATUS_SUCCESS)
                has_cleanup_failure = true;
        }
    }
    cudnnBackendDescriptor_t get() const {
        return value_;
    }
    void close() {
        checked(cudnnBackendDestroyDescriptor(std::exchange(value_, nullptr)),
                "destroy original source descriptor");
    }
    void finalize() const {
        checked(cudnnBackendFinalize(value_), "finalize backend descriptor");
    }
    void initialize() const {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        checked(cudnnBackendInitialize(value_), "one-argument backend initialize");
#pragma GCC diagnostic pop
    }

   private:
    cudnnBackendDescriptor_t value_{};
};

void set(const Descriptor& descriptor, cudnnBackendAttributeName_t name,
         cudnnBackendAttributeType_t type, std::int64_t count, const void* values) {
    checked(cudnnBackendSetAttribute(descriptor.get(), name, type, count, values),
            "set backend attribute");
}
struct TensorLayout {
    std::array<std::int64_t, 4> dimensions{}, strides{};
};
void configure_tensor(const Descriptor& descriptor, cudnnDataType_t type, std::int64_t uid,
                      const TensorLayout& layout) {
    const std::int64_t alignment = 16;
    const bool is_virtual = false;
    set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type);
    set(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
    set(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, layout.dimensions.data());
    set(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, layout.strides.data());
    set(descriptor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
    descriptor.finalize();
}
template <typename T, std::size_t Count>
struct Guarded {
    std::uint64_t before{0x123456789abcdef0ULL};
    std::array<T, Count> values{};
    std::uint64_t after{0xfedcba9876543210ULL};
    void check() const {
        expect(before == 0x123456789abcdef0ULL && after == 0xfedcba9876543210ULL,
               "native getter preserves outer guards");
    }
};
void check_tensor(const Descriptor& descriptor, cudnnDataType_t type, std::int64_t uid,
                  const TensorLayout& layout) {
    Guarded<std::int64_t, 6> result;
    result.values.fill(-901);
    std::int64_t count = -1;
    checked(cudnnBackendGetAttribute(descriptor.get(), CUDNN_ATTR_TENSOR_DIMENSIONS,
                                     CUDNN_TYPE_INT64, 6, &count, result.values.data()),
            "read copied tensor dimensions");
    result.check();
    expect(count == 4 && result.values[4] == -901 && result.values[5] == -901,
           "oversized capacity returns actual rank and leaves the unused tail untouched");
    for (std::size_t index = 0; index != 4; ++index)
        expect(result.values[index] == layout.dimensions[index], "copied tensor dimensions agree");
    result.values.fill(-903);
    checked(cudnnBackendGetAttribute(descriptor.get(), CUDNN_ATTR_TENSOR_DIMENSIONS,
                                     CUDNN_TYPE_INT64, 2, &count, result.values.data()),
            "bounded prefix tensor query");
    result.check();
    expect(count == 2, "data-present native query reports copied count at short capacity");
    for (std::size_t index = 0; index != result.values.size(); ++index)
        expect(result.values[index] == (index < 2 ? layout.dimensions[index] : -903),
               "short-capacity query writes exactly the requested prefix");
    result.values.fill(-902);
    checked(cudnnBackendGetAttribute(descriptor.get(), CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64,
                                     6, nullptr, result.values.data()),
            "read copied tensor strides without count output");
    result.check();
    expect(result.values[4] == -902 && result.values[5] == -902,
           "null count output does not authorize writing unused tail");
    for (std::size_t index = 0; index != 4; ++index)
        expect(result.values[index] == layout.strides[index], "copied tensor strides agree");
    std::int64_t observed_uid = -1;
    cudnnDataType_t observed_type = CUDNN_DATA_INT8;
    checked(cudnnBackendGetAttribute(descriptor.get(), CUDNN_ATTR_TENSOR_UNIQUE_ID,
                                     CUDNN_TYPE_INT64, 1, &count, &observed_uid),
            "read copied tensor UID");
    expect(count == 1 && observed_uid == uid, "native parent preserves original tensor UID");
    checked(cudnnBackendGetAttribute(descriptor.get(), CUDNN_ATTR_TENSOR_DATA_TYPE,
                                     CUDNN_TYPE_DATA_TYPE, 1, &count, &observed_type),
            "read copied tensor type");
    expect(count == 1 && observed_type == type, "native parent preserves tensor storage type");
    count = -1;
    checked(cudnnBackendGetAttribute(descriptor.get(), CUDNN_ATTR_TENSOR_DIMENSIONS,
                                     CUDNN_TYPE_INT64, 0, &count, nullptr),
            "count-only tensor query");
    expect(count == 4, "count-only native query returns complete rank");
}
void copy_attribute(const Descriptor& source, cudnnBackendAttributeName_t name,
                    const Descriptor& destination) {
    Guarded<cudnnBackendDescriptor_t, 1> output;
    output.values[0] = destination.get();
    std::int64_t count = -1;
    checked(cudnnBackendGetAttribute(source.get(), name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count,
                                     output.values.data()),
            "copy native attribute into existing descriptor");
    output.check();
    expect(count == 1 && output.values[0] == destination.get(),
           "descriptor-valued get mutates destination without replacing its public handle");
}
void reduction_copy(cudnnDataType_t type) {
    constexpr TensorLayout input_layout{{1, 2, 3, 4}, {24, 12, 4, 1}};
    constexpr TensorLayout output_layout{{1, 2, 1, 1}, {2, 1, 1, 1}};
    Descriptor input(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    Descriptor output(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    Descriptor reduction(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
    Descriptor operation(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    configure_tensor(input, type, 7, input_layout);
    configure_tensor(output, type, 8, output_layout);
    const auto mode = CUDNN_REDUCE_TENSOR_ADD;
    set(reduction, CUDNN_ATTR_REDUCTION_OPERATOR, CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &mode);
    set(reduction, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type);
    reduction.finalize();
    for (const auto& [name, descriptor] :
         std::array{std::pair{CUDNN_ATTR_OPERATION_REDUCTION_XDESC, input.get()},
                    std::pair{CUDNN_ATTR_OPERATION_REDUCTION_YDESC, output.get()},
                    std::pair{CUDNN_ATTR_OPERATION_REDUCTION_DESC, reduction.get()}})
        set(operation, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &descriptor);

    // Reset and reconfigure the original object after the parent accepted it.
    // A snapshot must retain the old values, not a live source-ID dependency.
    input.initialize();
    configure_tensor(input, type, 107, output_layout);
    input.close();
    output.close();
    reduction.close();
    operation.finalize();

    Descriptor copied_input(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    Descriptor copied_output(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    Descriptor copied_reduction(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
    copy_attribute(operation, CUDNN_ATTR_OPERATION_REDUCTION_XDESC, copied_input);
    copy_attribute(operation, CUDNN_ATTR_OPERATION_REDUCTION_YDESC, copied_output);
    copy_attribute(operation, CUDNN_ATTR_OPERATION_REDUCTION_DESC, copied_reduction);
    // Successful copies remain independently usable even after source removal.
    operation.close();
    check_tensor(copied_input, type, 7, input_layout);
    check_tensor(copied_output, type, 8, output_layout);
    cudnnReduceTensorOp_t observed = CUDNN_REDUCE_TENSOR_MAX;
    std::int64_t count = -1;
    checked(cudnnBackendGetAttribute(copied_reduction.get(), CUDNN_ATTR_REDUCTION_OPERATOR,
                                     CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &count, &observed),
            "read independent copied reduction");
    expect(count == 1 && observed == mode, "copied reduction outlives source operation");
    cudnnDataType_t observed_precision = CUDNN_DATA_INT8;
    checked(cudnnBackendGetAttribute(copied_reduction.get(), CUDNN_ATTR_REDUCTION_COMP_TYPE,
                                     CUDNN_TYPE_DATA_TYPE, 1, &count, &observed_precision),
            "read independent copied reduction precision");
    expect(count == 1 && observed_precision == type, "copied reduction precision is preserved");
    std::cout << "backend_metadata_case;type=" << static_cast<int>(type)
              << ";source_reinitialized=1;source_destroyed=1;destination_handles_preserved=1"
                 ";source_snapshot_preserved=1;unused_output_tail_preserved=1\n";
}
}  // namespace

int main() {
    try {
        const auto version = cudnnGetVersion();
        ++calls;
        expect(version == 91002, "metadata probe is bound to audited native 9.10.2");
        reduction_copy(CUDNN_DATA_FLOAT);
        reduction_copy(CUDNN_DATA_DOUBLE);
        expect(!has_cleanup_failure, "all known owned metadata descriptors destroy successfully");
        std::cout << "backend_metadata_success;configurations=2;calls=" << calls
                  << ";cuda_contexts_requested=0;compute_calls=0;owned_route_qualified=0"
                     ";full_device_qualified=0;priority_qualified=0\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "backend_metadata_failed: " << error.what() << '\n';
        return 1;
    }
}
