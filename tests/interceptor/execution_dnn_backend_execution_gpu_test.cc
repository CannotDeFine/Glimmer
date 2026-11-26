#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Ordinary SDK input, not an automatic CTest. Compilation is not qualification.
// A separately authorized device run must check native and owned paths. These
// small correctness cases are not full-device utilization or priority evidence.
namespace {
constexpr std::size_t kGuard = 256, kMaxWorkspace = std::size_t{64} * 1024 * 1024;
bool has_failure{};
unsigned int execute_calls{};
void expect(bool condition, const char* message) {
    if (!condition) {
        has_failure = true;
        throw std::runtime_error(message);
    }
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status)) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&& other) noexcept : value_(std::exchange(other.value_, {})) {}
    ~Resource() {
        if (value_ && !has_failure && static_cast<int>(Destroy(std::exchange(value_, {}))))
            has_failure = true;
    }
    Handle get() const {
        return value_;
    }
    Handle* output() {
        return &value_;
    }
    void reset() {
        if (value_) checked(Destroy(std::exchange(value_, {})), "destroy original resource");
    }

   private:
    Handle value_{};
};
using Descriptor = Resource<cudnnBackendDescriptor_t, cudnnBackendDestroyDescriptor>;
using Context = Resource<CUcontext, cuCtxDestroy>;
using Handle = Resource<cudnnHandle_t, cudnnDestroy>;
using Stream = Resource<CUstream, cuStreamDestroy>;
struct RetainedProperty {
    Descriptor descriptor;
    unsigned int source_attribute{};
};
struct MetadataCase {
    cudnnBackendDescriptor_t property{};
    std::vector<RetainedProperty>* retained{};
};
unsigned int property_checks{}, layout_checks{}, note_checks{};

Descriptor create(cudnnBackendDescriptorType_t kind) {
    Descriptor result;
    checked(cudnnBackendCreateDescriptor(kind, result.output()), "create backend descriptor");
    return result;
}
void set(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name,
         cudnnBackendAttributeType_t type, std::int64_t count, const void* values) {
    checked(cudnnBackendSetAttribute(descriptor, name, type, count, values),
            "set backend attribute");
}
void finalize(cudnnBackendDescriptor_t descriptor) {
    checked(cudnnBackendFinalize(descriptor), "finalize backend descriptor");
}
std::int64_t integer(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name) {
    std::array<std::int64_t, 3> value{-91, -92, -93};
    std::int64_t count{-94};
    checked(cudnnBackendGetAttribute(descriptor, name, CUDNN_TYPE_INT64, 1, &count, &value[1]),
            "query original scalar attribute");
    expect(count == 1 && value.front() == -91 && value.back() == -93,
           "scalar query preserves adjacent values");
    return value[1];
}
std::uint64_t checksum(std::span<const std::byte> bytes) {
    std::uint64_t value = 14695981039346656037ULL;
    for (const auto byte : bytes) {
        value ^= std::to_integer<unsigned int>(byte);
        value *= 1099511628211ULL;
    }
    return value;
}
std::vector<std::byte> property_json(cudnnBackendDescriptor_t descriptor,
                                     unsigned int source_attribute, bool after_context) {
    constexpr std::int32_t marker = -1981;
    std::array<std::int32_t, 3> id{marker, marker, marker};
    std::int64_t count{-191};
    auto status = cudnnBackendGetAttribute(descriptor, CUDNN_ATTR_DEVICEPROP_DEVICE_ID,
                                           CUDNN_TYPE_INT32, 1, &count, &id[1]);
    if (status == CUDNN_STATUS_NOT_INITIALIZED) {
        expect(count == -191 && std::ranges::all_of(id, [](auto value) { return value == marker; }),
               "unfinalized property error preserves all ID and count bytes before retry");
        // A copied header need not be finalized. HANDLE/JSON-populated values
        // can still finalize without resurrecting the destroyed source context.
        finalize(descriptor);
        status = cudnnBackendGetAttribute(descriptor, CUDNN_ATTR_DEVICEPROP_DEVICE_ID,
                                          CUDNN_TYPE_INT32, 1, &count, &id[1]);
    }
    checked(status, "read copied device property ID");
    expect(count == 1 && id.front() == marker && id.back() == marker && id[1] == 0,
           "INT32 property ID uses exact four-byte output and the requested test device");
    std::int64_t bytes{-1};
    checked(cudnnBackendGetAttribute(descriptor, CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION,
                                     CUDNN_TYPE_CHAR, 0, &bytes, nullptr),
            "observe exact property JSON length");
    expect(bytes > 1 && bytes <= 65536, "native property JSON fits the explicit diagnostic bound");
    std::vector<std::byte> guarded(static_cast<std::size_t>(bytes) + 16, std::byte{0xa9});
    count = -191;
    const auto short_status =
        cudnnBackendGetAttribute(descriptor, CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION,
                                 CUDNN_TYPE_CHAR, bytes - 1, &count, guarded.data() + 8);
    expect(short_status == CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT && count == -191 &&
               std::ranges::all_of(guarded, [](auto value) { return value == std::byte{0xa9}; }),
           "short positive JSON capacity preserves native error and every caller byte/count");
    checked(cudnnBackendGetAttribute(descriptor, CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION,
                                     CUDNN_TYPE_CHAR, bytes + 8, &count, guarded.data() + 8),
            "read complete JSON without adding a terminator");
    expect(count == bytes &&
               std::ranges::all_of(std::span{guarded}.first(8),
                                   [](auto value) { return value == std::byte{0xa9}; }) &&
               std::ranges::all_of(std::span{guarded}.last(8),
                                   [](auto value) { return value == std::byte{0xa9}; }),
           "JSON writes exactly its native byte length, no NUL or unused-capacity suffix");
    std::int64_t negative{-1};
    checked(cudnnBackendGetAttribute(descriptor, CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION,
                                     CUDNN_TYPE_CHAR, -1, &negative, nullptr),
            "negative count-only JSON query");
    expect(negative == bytes, "JSON count-only signed request does not allocate from capacity");
    const auto output = std::span{guarded}.subspan(8, static_cast<std::size_t>(bytes));
    std::cout << "backend_property_case;source=" << source_attribute
              << ";after_context=" << after_context << ";device_id=" << id[1] << ";bytes=" << bytes
              << ";checksum=" << checksum(output) << '\n';
    ++property_checks;
    return {output.begin(), output.end()};
}
Descriptor handle_property(cudnnHandle_t handle) {
    auto result = create(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR);
    set(result.get(), CUDNN_ATTR_DEVICEPROP_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
    return result;
}
void retain_property(cudnnBackendDescriptor_t source, cudnnBackendAttributeName_t name,
                     std::vector<RetainedProperty>& retained) {
    auto destination = create(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR);
    std::array<cudnnBackendDescriptor_t, 3> pointers{source, destination.get(), source};
    std::int64_t count{-1};
    checked(cudnnBackendGetAttribute(source, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count,
                                     &pointers[1]),
            "copy independent device property into precreated destination");
    expect(count == 1 && pointers[0] == source && pointers[1] == destination.get() &&
               pointers[2] == source,
           "property copy never replaces caller handles or adjacent pointer guards");
    retained.push_back({std::move(destination), static_cast<unsigned int>(name)});
}
void enum_array(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name,
                cudnnBackendAttributeType_t type, std::uint32_t limit) {
    std::int64_t count{-1};
    checked(cudnnBackendGetAttribute(descriptor, name, type, 0, &count, nullptr),
            "observe native enum-array cardinality");
    expect(count >= 0 && count <= limit, "native enum array has its own fixed bound");
    constexpr std::uint32_t marker = 0xa8a8a8a8U;
    std::vector<std::uint32_t> values(static_cast<std::size_t>(count) + 3, marker);
    const auto expected = count;
    checked(cudnnBackendGetAttribute(descriptor, name, type, 0, &count, values.data() + 1),
            "zero-request native enum query emits all values");
    expect(count == expected && values.front() == marker &&
               values[static_cast<std::size_t>(count) + 1] == marker && values.back() == marker,
           "zero request is not a zero-byte grant and preserves the exact output tail");
    for (std::size_t index = 0; index < static_cast<std::size_t>(count); ++index)
        expect(values[index + 1] < limit && (!index || values[index] < values[index + 1]),
               "native bitmap traversal emits valid strictly increasing raw enum bits");
    const auto original = values;
    checked(cudnnBackendGetAttribute(descriptor, name, type, 0, nullptr, values.data() + 1),
            "native enum output permits an omitted count");
    expect(values == original, "optional count does not alter native enum output");
    std::cout << "backend_enum_case;attribute=" << static_cast<unsigned int>(name)
              << ";count=" << count << ";checksum="
              << checksum(
                     std::as_bytes(std::span{values}.subspan(1, static_cast<std::size_t>(count))))
              << '\n';
    ++note_checks;
}
void engine_metadata(cudnnBackendDescriptor_t engine, std::vector<RetainedProperty>& retained) {
    enum_array(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, 10);
    enum_array(engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE, 4);
    std::int64_t count{-1};
    checked(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_LAYOUT_INFO,
                                     CUDNN_TYPE_BACKEND_DESCRIPTOR, INT64_MAX, &count, nullptr),
            "observe generated LayoutInfo count with no capacity-sized allocation");
    expect(count >= 0 && count <= 65534, "native layout count leaves two valid tail targets");
    const auto expected = count;
    std::vector<Descriptor> destinations;
    std::vector<cudnnBackendDescriptor_t> pointers;
    for (std::int64_t index = 0; index < expected + 2; ++index) {
        destinations.push_back(create(CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR));
        pointers.push_back(destinations.back().get());
    }
    const auto original = pointers;
    checked(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_LAYOUT_INFO,
                                     CUDNN_TYPE_BACKEND_DESCRIPTOR, expected + 2, &count,
                                     pointers.data()),
            "populate layout prefix with a fully validated unused destination tail");
    expect(count == expected && pointers == original,
           "generated layout emission preserves every caller identity and unused tail");
    for (std::int64_t index = count; index < expected + 2; ++index) {
        std::int64_t untouched{-1981};
        expect(cudnnBackendGetAttribute(pointers[static_cast<std::size_t>(index)],
                                        CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID, CUDNN_TYPE_INT64, 0,
                                        &untouched, nullptr) == CUDNN_STATUS_NOT_INITIALIZED &&
                   untouched == -1981,
               "unused LayoutInfo tail retains its unfinalized header and untouched count");
    }
    for (std::int64_t index = 0; index < count; ++index) {
        std::int64_t uid{}, written{};
        auto status = cudnnBackendGetAttribute(pointers[static_cast<std::size_t>(index)],
                                               CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID, CUDNN_TYPE_INT64,
                                               1, &written, &uid);
        if (status == CUDNN_STATUS_NOT_INITIALIZED) {
            finalize(pointers[static_cast<std::size_t>(index)]);
            status = cudnnBackendGetAttribute(pointers[static_cast<std::size_t>(index)],
                                              CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID, CUDNN_TYPE_INT64,
                                              1, &written, &uid);
        }
        checked(status, "read actual copied LayoutInfo UID");
        expect(written == 1, "copied LayoutInfo has an actual complete native UID");
        enum_array(pointers[static_cast<std::size_t>(index)], CUDNN_ATTR_LAYOUT_INFO_TYPES,
                   CUDNN_TYPE_LAYOUT_TYPE, 4);
        std::cout << "backend_layout_case;index=" << index << ";uid=" << uid << '\n';
        ++layout_checks;
    }
    retain_property(engine, CUDNN_ATTR_ENGINE_DEVICEPROP, retained);
}
void attach_property(cudnnBackendDescriptor_t parent, cudnnBackendAttributeName_t name,
                     MetadataCase metadata) {
    if (metadata.property) set(parent, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &metadata.property);
}
Descriptor tensor(std::int64_t uid, const std::array<std::int64_t, 4>& dimensions) {
    auto result = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    std::array<std::int64_t, 4> strides{1, 1, 1, 1};
    for (std::size_t i = 3; i != 0; --i) strides[i - 1] = strides[i] * dimensions[i];
    constexpr std::int64_t alignment = 16;
    constexpr auto type = CUDNN_DATA_FLOAT;
    set(result.get(), CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type);
    set(result.get(), CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dimensions.data());
    set(result.get(), CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides.data());
    set(result.get(), CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
    set(result.get(), CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    finalize(result.get());
    return result;
}
class Device final {
   public:
    explicit Device(std::size_t bytes)
        : bytes_(bytes), initial_(bytes + 2 * kGuard, std::byte{0xa7}) {
        expect(bytes <= kMaxWorkspace, "bound correctness input storage");
        checked(cuMemAlloc(&base_, initial_.size()), "allocate guarded input");
        reset(std::vector<std::byte>(bytes));
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() {
        if (base_ && !has_failure && cuMemFree(base_) != CUDA_SUCCESS) has_failure = true;
    }
    void* get() const {
        // NOLINTNEXTLINE(performance-no-int-to-ptr): CUDA SDK device-pointer ABI.
        return reinterpret_cast<void*>(base_ + kGuard);
    }
    void reset(std::span<const std::byte> bytes) {
        expect(bytes.size() == bytes_, "reset complete input payload");
        std::copy(bytes.begin(), bytes.end(), initial_.begin() + kGuard);
        checked(cuMemcpyHtoD(base_, initial_.data(), initial_.size()), "upload guarded input");
    }
    std::vector<std::byte> read() const {
        std::vector<std::byte> result(initial_.size());
        checked(cuMemcpyDtoH(result.data(), base_, result.size()), "download guarded output");
        expect(std::equal(result.begin(), result.begin() + kGuard, initial_.begin()) &&
                   std::equal(result.end() - kGuard, result.end(), initial_.end() - kGuard),
               "every prefix and suffix guard byte is unchanged");
        return {result.begin() + kGuard, result.end() - kGuard};
    }
    void unchanged() const {
        const auto result = read();
        expect(std::equal(result.begin(), result.end(), initial_.begin() + kGuard),
               "complete read-only input is unchanged");
    }

   private:
    CUdeviceptr base_{};
    std::size_t bytes_{};
    std::vector<std::byte> initial_;
};
struct Shape {
    int batch, channels, height, width, outputs, rows, columns;
    int pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
    int output_h() const {
        return (height + 2 * pad_h - dilation_h * (rows - 1) - 1) / stride_h + 1;
    }
    int output_w() const {
        return (width + 2 * pad_w - dilation_w * (columns - 1) - 1) / stride_w + 1;
    }
};
struct Model {
    Shape shape;
    std::vector<float> x, w, initial;
    std::vector<double> reference;
};
Model model(Shape shape) {
    Model result{shape,
                 std::vector<float>(static_cast<std::size_t>(shape.batch) * shape.channels *
                                    shape.height * shape.width),
                 std::vector<float>(static_cast<std::size_t>(shape.outputs) * shape.channels *
                                    shape.rows * shape.columns),
                 std::vector<float>(static_cast<std::size_t>(shape.batch) * shape.outputs *
                                    shape.output_h() * shape.output_w()),
                 {}};
    for (std::size_t i = 0; i != result.x.size(); ++i)
        result.x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 32;
    for (std::size_t i = 0; i != result.w.size(); ++i)
        result.w[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 16;
    for (std::size_t i = 0; i != result.initial.size(); ++i)
        result.initial[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 8;
    // Independent NCHW cross-correlation; no SDK/plan layout is used by this oracle.
    for (int n = 0; n != shape.batch; ++n)
        for (int k = 0; k != shape.outputs; ++k)
            for (int y = 0; y != shape.output_h(); ++y)
                for (int x = 0; x != shape.output_w(); ++x) {
                    double sum{};
                    for (int c = 0; c != shape.channels; ++c)
                        for (int r = 0; r != shape.rows; ++r)
                            for (int s = 0; s != shape.columns; ++s) {
                                const int iy =
                                    y * shape.stride_h - shape.pad_h + r * shape.dilation_h;
                                const int ix =
                                    x * shape.stride_w - shape.pad_w + s * shape.dilation_w;
                                if (iy < 0 || iy >= shape.height || ix < 0 || ix >= shape.width)
                                    continue;
                                const auto input =
                                    ((n * shape.channels + c) * shape.height + iy) * shape.width +
                                    ix;
                                const auto weight =
                                    ((k * shape.channels + c) * shape.rows + r) * shape.columns + s;
                                sum += static_cast<double>(result.x[input]) * result.w[weight];
                            }
                    result.reference.push_back(sum);
                }
    return result;
}
Descriptor plan(cudnnBackendDescriptor_t graph, std::int64_t& selected,
                MetadataCase metadata = {}) {
    const auto count = integer(graph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT);
    expect(count > 0 && count <= 4096, "bounded original engine enumeration is nonempty");
    for (std::int64_t index = 0; index != count; ++index) {
        expect(!has_failure, "previous candidate cleanup succeeded");
        auto engine = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        set(engine.get(), CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
            &graph);
        set(engine.get(), CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &index);
        attach_property(engine.get(), CUDNN_ATTR_ENGINE_DEVICEPROP, metadata);
        if (cudnnBackendFinalize(engine.get()) != CUDNN_STATUS_SUCCESS) continue;
        auto config = create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        auto native_engine = engine.get();
        set(config.get(), CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
            &native_engine);
        if (cudnnBackendFinalize(config.get()) != CUDNN_STATUS_SUCCESS) continue;
        auto result = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        auto native_config = config.get();
        set(result.get(), CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
            &native_config);
        attach_property(result.get(), CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP, metadata);
        if (cudnnBackendFinalize(result.get()) != CUDNN_STATUS_SUCCESS) continue;
        const auto workspace = integer(result.get(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
        expect(workspace >= 0, "original workspace size is nonnegative");
        if (static_cast<std::uint64_t>(workspace) > kMaxWorkspace) continue;
        if (metadata.retained) {
            engine_metadata(engine.get(), *metadata.retained);
            retain_property(result.get(), CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP, *metadata.retained);
        }
        selected = index;
        return result;
    }
    throw std::runtime_error(
        "no executable original plan within fixture bounds; qualification failed");
}
std::int64_t descriptor_count(cudnnBackendDescriptor_t descriptor,
                              cudnnBackendAttributeName_t name) {
    std::int64_t count{-1};
    // Config's normalized table query returns zero for request zero; its fixed
    // 43-entry bound is different from Heur/Engine count-only semantics.
    const std::int64_t requested = name == CUDNN_ATTR_ENGINECFG_KNOB_CHOICES ? 43 : 0;
    checked(cudnnBackendGetAttribute(descriptor, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, requested,
                                     &count, nullptr),
            "query generated descriptor capacity");
    expect(count >= 0 && count <= 4096, "generated descriptor count fits diagnostic bound");
    return count;
}
std::vector<Descriptor> generated(cudnnBackendDescriptor_t source, cudnnBackendAttributeName_t name,
                                  cudnnBackendDescriptorType_t kind) {
    const auto capacity = descriptor_count(source, name);
    std::vector<Descriptor> outputs;
    std::vector<cudnnBackendDescriptor_t> pointers;
    outputs.reserve(static_cast<std::size_t>(capacity));
    pointers.reserve(static_cast<std::size_t>(capacity));
    for (std::int64_t i = 0; i != capacity; ++i) {
        outputs.push_back(create(kind));
        pointers.push_back(outputs.back().get());
    }
    if (!capacity) return outputs;
    const auto original = pointers;
    std::int64_t emitted{-1};
    checked(cudnnBackendGetAttribute(source, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, capacity,
                                     &emitted, pointers.data()),
            "populate precreated generated descriptor destinations");
    expect(emitted >= 0 && emitted <= capacity && pointers == original,
           "generated prefix is bounded and caller descriptor identities are unchanged");
    // Native heuristics can filter candidates: only its actual emitted prefix
    // contains finalized Configs, not every count-only capacity slot.
    outputs.resize(static_cast<std::size_t>(emitted));
    return outputs;
}
Descriptor heuristic_plan(Descriptor& graph, std::int64_t& selected, MetadataCase metadata = {}) {
    auto heuristic = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    auto native_graph = graph.get();
    constexpr auto mode = CUDNN_HEUR_MODE_INSTANT;
    set(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
        &native_graph);
    set(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode);
    attach_property(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_DEVICEPROP, metadata);
    finalize(heuristic.get());
    if (metadata.retained)
        retain_property(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_DEVICEPROP, *metadata.retained);
    // The heuristic owns a copied graph, not a live application source token.
    graph.reset();
    auto configs = generated(heuristic.get(), CUDNN_ATTR_ENGINEHEUR_RESULTS,
                             CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    expect(!configs.empty(), "native heuristic emits at least one executable candidate");
    heuristic.reset();
    for (const auto& config : configs) {
        expect(!has_failure, "previous generated candidate cleanup succeeded");
        auto engine = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        auto native_engine = engine.get();
        std::int64_t count{-1};
        checked(cudnnBackendGetAttribute(config.get(), CUDNN_ATTR_ENGINECFG_ENGINE,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &native_engine),
                "copy generated configuration engine");
        expect(count == 1 && native_engine == engine.get(), "generated engine identity");
        const auto index = integer(engine.get(), CUDNN_ATTR_ENGINE_GLOBAL_INDEX);
        auto copied_graph = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        auto native_copy = copied_graph.get();
        checked(cudnnBackendGetAttribute(engine.get(), CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, &native_copy),
                "copy generated engine graph after source destruction");
        expect(count == 1 && native_copy == copied_graph.get() &&
                   integer(copied_graph.get(), CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT) > 0,
               "generated engine retains its original graph semantics");
        auto info = generated(engine.get(), CUDNN_ATTR_ENGINE_KNOB_INFO,
                              CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR);
        for (const auto& knob : info) {
            const auto minimum = integer(knob.get(), CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE);
            const auto maximum = integer(knob.get(), CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE);
            const auto stride = integer(knob.get(), CUDNN_ATTR_KNOB_INFO_STRIDE);
            expect(minimum <= maximum && stride > 0, "native generated knob limits");
        }
        auto choices = generated(config.get(), CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                 CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
        for (const auto& knob : choices)
            (void)integer(knob.get(), CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE);
        auto result = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        auto native_config = config.get();
        set(result.get(), CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
            &native_config);
        attach_property(result.get(), CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP, metadata);
        if (cudnnBackendFinalize(result.get()) != CUDNN_STATUS_SUCCESS) continue;
        const auto workspace = integer(result.get(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
        expect(workspace >= 0, "heuristic plan workspace is nonnegative");
        if (static_cast<std::uint64_t>(workspace) > kMaxWorkspace) continue;
        if (metadata.retained) {
            engine_metadata(engine.get(), *metadata.retained);
            retain_property(result.get(), CUDNN_ATTR_EXECUTION_PLAN_DEVICEPROP, *metadata.retained);
        }
        selected = index;
        return result;
    }
    throw std::runtime_error("no heuristic-generated plan fits diagnostic bounds");
}
void run(cudnnHandle_t handle, CUstream stream, Shape shape, unsigned int configuration,
         bool uses_heuristics, MetadataCase metadata = {}) {
    expect(!has_failure, "previous configuration cleanup succeeded");
    const auto data = model(shape);
    auto x = tensor(101, {shape.batch, shape.channels, shape.height, shape.width});
    auto w = tensor(103, {shape.outputs, shape.channels, shape.rows, shape.columns});
    auto y = tensor(107, {shape.batch, shape.outputs, shape.output_h(), shape.output_w()});
    auto convolution = create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    constexpr auto precision = CUDNN_DATA_FLOAT;
    constexpr auto mode = CUDNN_CROSS_CORRELATION;
    constexpr std::int64_t spatial = 2;
    const std::array<std::int64_t, 2> padding{shape.pad_h, shape.pad_w};
    const std::array<std::int64_t, 2> strides{shape.stride_h, shape.stride_w};
    const std::array<std::int64_t, 2> dilations{shape.dilation_h, shape.dilation_w};
    set(convolution.get(), CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &precision);
    set(convolution.get(), CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
    set(convolution.get(), CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
    set(convolution.get(), CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2,
        padding.data());
    set(convolution.get(), CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2,
        padding.data());
    set(convolution.get(), CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2,
        strides.data());
    set(convolution.get(), CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dilations.data());
    finalize(convolution.get());
    auto operation = create(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    const std::array<cudnnBackendDescriptor_t, 4> children{x.get(), w.get(), y.get(),
                                                           convolution.get()};
    constexpr std::array names{CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                               CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                               CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                               CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC};
    for (std::size_t i = 0; i != names.size(); ++i)
        set(operation.get(), names[i], CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &children[i]);
    constexpr float alpha = 1.25F, beta = 0.5F;
    set(operation.get(), CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1,
        &alpha);
    set(operation.get(), CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
    finalize(operation.get());
    auto graph = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    // Exact configured deprecated numeric attribute, not a changed SDK ABI.
    set(graph.get(), static_cast<cudnnBackendAttributeName_t>(800), CUDNN_TYPE_HANDLE, 1, &handle);
    auto original_operation = operation.get();
    set(graph.get(), CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
        &original_operation);
    finalize(graph.get());
    // Original graph snapshots must survive destruction/reset of the sources.
    checked(cudnnBackendInitialize(x.get()), "reset copied original tensor");
    x.reset();
    w.reset();
    y.reset();
    convolution.reset();
    operation.reset();
    std::int64_t engine{-1};
    auto execution_plan = uses_heuristics ? heuristic_plan(graph, engine, metadata)
                                          : plan(graph.get(), engine, metadata);
    expect(!has_failure, "selected plan source cleanup succeeded");
    graph.reset();
    const auto workspace_bytes =
        integer(execution_plan.get(), CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE);
    expect(workspace_bytes >= 0 && static_cast<std::uint64_t>(workspace_bytes) <= kMaxWorkspace,
           "selected original workspace fits exact fixture bound");
    Device input(data.x.size() * sizeof(float)), weights(data.w.size() * sizeof(float));
    Device output(data.initial.size() * sizeof(float)),
        workspace(static_cast<std::size_t>(workspace_bytes));
    input.reset(std::as_bytes(std::span{data.x}));
    weights.reset(std::as_bytes(std::span{data.w}));
    auto variant = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    constexpr std::array<std::int64_t, 3> uids{107, 101, 103};
    const std::array<void*, 3> pointers{output.get(), input.get(), weights.get()};
    void* workspace_pointer = workspace_bytes ? workspace.get() : nullptr;
    set(variant.get(), CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids.data());
    set(variant.get(), CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3,
        pointers.data());
    set(variant.get(), CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1,
        &workspace_pointer);
    finalize(variant.get());
    for (unsigned int repeat = 0; repeat != 2; ++repeat) {
        output.reset(std::as_bytes(std::span{data.initial}));
        checked(cuCtxSynchronize(), "order pageable uploads before nonblocking execution");
        checked(cudnnBackendExecute(handle, execution_plan.get(), variant.get()),
                "execute original graph");
        ++execute_calls;
        checked(cuStreamSynchronize(stream), "prove original graph completion");
        const auto observed = output.read();
        input.unchanged();
        weights.unchanged();
        (void)workspace.read();
        std::cout << "backend_execution_case;configuration=" << configuration
                  << ";repeat=" << repeat << ";engine=" << engine
                  << ";workspace=" << workspace_bytes << ";values=";
        for (std::size_t i = 0; i != data.reference.size(); ++i) {
            float value{};
            std::memcpy(&value, observed.data() + i * sizeof(float), sizeof(float));
            const auto expected = alpha * data.reference[i] + beta * data.initial[i];
            expect(std::isfinite(value) &&
                       std::abs(value - expected) <= 2e-5 + 2e-4 * std::abs(expected),
                   "every output agrees with independent convolution including nonzero beta");
            std::cout << value << ',';
        }
        std::cout << '\n';
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const bool uses_heuristics = argc == 2 && std::string(argv[1]) == "--heuristics";
        const bool uses_properties = argc == 2 && std::string(argv[1]) == "--engine-properties";
        expect(argc == 1 || uses_heuristics || uses_properties,
               "expected optional --heuristics or --engine-properties");
        std::cout << std::setprecision(9);
        std::vector<RetainedProperty> retained;
        std::vector<std::byte> original_json;
        {
            checked(cuInit(0), "initialize Driver");
            CUdevice device{};
            checked(cuDeviceGet(&device, 0), "select test device");
            Context context;
            checked(cuCtxCreate(context.output(), 0, device), "create owned context");
            Stream stream;
            checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING),
                    "create original stream");
            Handle handle;
            checked(cudnnCreate(handle.output()), "create original cuDNN handle");
            checked(cudnnSetStream(handle.get(), stream.get()), "bind original stream");
            std::optional<Descriptor> populated, restored;
            if (uses_properties) {
                auto discovered = create(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR);
                finalize(discovered.get());
                original_json = property_json(discovered.get(), 2700, false);
                populated.emplace(handle_property(handle.get()));
                finalize(populated->get());
                restored.emplace(create(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR));
                set(restored->get(), CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION, CUDNN_TYPE_CHAR,
                    static_cast<std::int64_t>(original_json.size()), original_json.data());
                finalize(restored->get());
                retained.push_back({std::move(discovered), 2700});
            }
            run(handle.get(), stream.get(), {2, 4, 5, 6, 3, 3, 2, 1, 0, 1, 2, 1, 1}, 0,
                uses_heuristics || uses_properties,
                {populated ? populated->get() : nullptr, uses_properties ? &retained : nullptr});
            run(handle.get(), stream.get(), {1, 3, 7, 7, 4, 2, 3, 1, 2, 2, 1, 2, 2}, 1,
                uses_heuristics,
                {restored ? restored->get() : nullptr, uses_properties ? &retained : nullptr});
            checked(cuStreamSynchronize(stream.get()), "final physical completion");
            if (uses_properties) {
                auto pending_handle = handle_property(handle.get());
                auto pending_json = create(CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR);
                set(pending_json.get(), CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION, CUDNN_TYPE_CHAR,
                    static_cast<std::int64_t>(original_json.size()), original_json.data());
                retained.push_back({std::move(pending_handle), 2701});
                retained.push_back({std::move(pending_json), 2702});
            }
        }
        for (const auto& property : retained) {
            const auto copied =
                property_json(property.descriptor.get(), property.source_attribute, true);
            if (property.source_attribute == 2700 || property.source_attribute == 2702)
                expect(copied == original_json,
                       "discovered and restored JSON bytes survive source context destruction");
        }
        retained.clear();
        expect(!has_failure && execute_calls == 4, "complete executions and all original cleanup");
        std::cout << "backend_execution_success;configurations=2;execute_calls=4;"
                     "full_device_qualified=0;priority_qualified=0\n";
        if (uses_heuristics)
            std::cout << "backend_heuristics_success;generated_plans=2;execute_calls=4;"
                         "full_device_qualified=0;priority_qualified=0\n";
        if (uses_properties) {
            expect(property_checks == 9 && note_checks == 4 + layout_checks,
                   "both planning paths and every retained property completed");
            std::cout << "backend_engine_properties_success;property_checks=" << property_checks
                      << ";layout_checks=" << layout_checks << ";enum_checks=" << note_checks
                      << ";execute_calls=4;full_device_qualified=0;priority_qualified=0\n";
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
