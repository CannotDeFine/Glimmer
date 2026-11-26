#pragma once

#include "dnn_library_info_fixture.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <thread>

// Independent host-only metadata model. Children are value copies, not native
// source pointers. This is an ABI/ownership oracle, not vendor graph execution.
namespace glimmer::tests::dnn_backend_graph {
using namespace backend::cuda;
namespace foundation = glimmer::tests::dnn_library_info;
namespace base = glimmer::tests::dnn;
using Native = cudnnBackendDescriptor_t;
struct Snapshot;
struct Attribute {
    std::uint32_t type{};
    std::int64_t count{};
    std::vector<std::byte> bytes;
    std::vector<std::shared_ptr<const Snapshot>> children;
};
struct Snapshot {
    std::uint32_t kind{};
    bool is_finalized{};
    std::map<std::uint32_t, Attribute> attributes;
};
struct Object {
    bool alive{};
    unsigned int destroy_attempts{};
    Snapshot snapshot;
};
struct State {
    std::array<Object, 128> objects{};
    unsigned int creates{}, destroys{}, initializes{}, finalizes{}, sets{}, gets{}, versions{};
    std::uint32_t last_name{}, last_type{};
    std::int64_t last_count{};
    bool last_has_array{}, last_has_count{};
    Native last_source{}, last_destination{};
    bool capture_set_logging{};
    std::vector<std::byte> logged_set_bytes;
    std::function<void()> callback;
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, destroy_status{CUDNN_STATUS_SUCCESS};
    bool null_create{}, partial_create{}, duplicate_create{};
};
struct Control {
    std::atomic<int> set_status{}, get_status{}, initialize_status{}, finalize_status{};
    std::atomic<std::size_t> version{91002};
    std::atomic<unsigned int> callbacks{};
    std::atomic<bool> pause{}, entered{}, release{true};
    std::atomic<bool> corrupt_count{}, corrupt_enum{}, corrupt_bool{};
    std::atomic<std::uint32_t> fail_get_name{};
};
inline State state;
inline Control control;
inline void reset() {
    foundation::reset();
    state = {};
    control.set_status.store(CUDNN_STATUS_SUCCESS);
    control.get_status.store(CUDNN_STATUS_SUCCESS);
    control.initialize_status.store(CUDNN_STATUS_SUCCESS);
    control.finalize_status.store(CUDNN_STATUS_SUCCESS);
    control.version.store(91002);
    control.callbacks.store(0);
    control.pause.store(false);
    control.entered.store(false);
    control.release.store(true);
    control.corrupt_count.store(false);
    control.corrupt_enum.store(false);
    control.corrupt_bool.store(false);
    control.fail_get_name.store(0);
}
inline void callback() {
    base::owner();
    ++control.callbacks;
    if (state.callback) state.callback();
    if (control.pause.exchange(false)) {
        control.entered.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!control.release.load()) {
            base::require(std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        }
    }
}
inline Object& object(Native native) {
    base::owner();
    for (auto& value : state.objects)
        if (&value == native) {
            base::require(value.alive);
            return value;
        }
    throw std::runtime_error("unknown native backend graph descriptor");
}
inline std::size_t width(std::uint32_t type) {
    if (type == 2 || type == 24) return 1;
    if (type == 3 || type == 5) return 8;
    if (type == 26) return 16;
    return 4;
}
template <typename T>
inline std::vector<std::byte> bytes(std::span<const T> values) {
    const auto input = std::as_bytes(values);
    return {input.begin(), input.end()};
}
template <typename T>
inline std::vector<std::byte> scalar(T value) {
    return bytes(std::span<const T>{&value, 1});
}
inline std::uint32_t child_kind(std::uint32_t name) {
    if (name == 1612) return 21;
    if (name == 750) return 0;
    if (name == 1523) return 18;
    if (name == 702 || name == 708 || name == 714) return 1;
    return 17;
}
inline bool is_child(std::uint32_t name) {
    return (name >= 1610 && name <= 1612) || (name >= 1520 && name <= 1523) ||
           (name >= 1525 && name <= 1527) || (name >= 750 && name <= 753) ||
           (name >= 756 && name <= 758) || (name >= 702 && name <= 705) ||
           (name >= 708 && name <= 711) || (name >= 714 && name <= 717);
}
inline Attribute default_attribute(std::uint32_t kind, std::uint32_t name) {
    if (kind == 17) {
        if (name == 901) return {1, 1, scalar(std::uint32_t{0}), {}};
        if (name == 909) return {20, 1, scalar(std::uint32_t{0}), {}};
        if (name == 907 || name == 908) return {2, 1, scalar(false), {}};
        if (name == 902 || name == 903) return {3, 0, {}, {}};
        if (name == 904) return {3, 1, scalar(std::int64_t{1}), {}};
        if (name == 905 || name == 906) return {3, 1, scalar(std::int64_t{-1}), {}};
        return {3, 1, scalar(std::int64_t{0}), {}};
    }
    if (kind == 0 && name == 0) return {14, 1, scalar(std::uint32_t{0}), {}};
    if (kind == 0 && name >= 3 && name <= 8) return {5, 1, scalar(0.0), {}};
    if (kind == 1 && name == 101) return {7, 1, scalar(std::uint32_t{1}), {}};
    if (kind == 1 && name >= 102 && name <= 105) return {3, 0, {}, {}};
    if (kind == 1 && name == 106) return {3, 1, scalar(std::int64_t{0}), {}};
    if (kind == 18 && name == 1503) return {5, 1, scalar(0.0), {}};
    if (kind == 21 && name == 1600) return {18, 1, scalar(std::uint32_t{0}), {}};
    return {1, 1, scalar(std::uint32_t{0}), {}};
}
inline CudaDnnDispatch dispatch(unsigned int missing = 0) {
    auto sdk = foundation::dispatch();
    sdk.version = [] {
        callback();
        ++state.versions;
        return control.version.load();
    };
    sdk.backend_create = [](cudnnBackendDescriptorType_t kind, Native* output) {
        callback();
        ++state.creates;
        if (!output) return CUDNN_STATUS_BAD_PARAM;
        *output = nullptr;
        if (state.null_create ||
            (state.create_status != CUDNN_STATUS_SUCCESS && !state.partial_create))
            return state.create_status;
        if (state.duplicate_create)
            for (auto& entry : state.objects)
                if (entry.alive) {
                    *output = &entry;
                    return state.create_status;
                }
        for (auto& entry : state.objects)
            if (!entry.alive) {
                entry = {};
                entry.alive = true;
                entry.snapshot.kind = static_cast<std::uint32_t>(kind);
                *output = &entry;
                return state.create_status;
            }
        return CUDNN_STATUS_ALLOC_FAILED;
    };
    sdk.backend_destroy = [](Native native) {
        auto& entry = object(native);
        callback();
        ++state.destroys;
        base::require(++entry.destroy_attempts == 1);
        if (state.destroy_status == CUDNN_STATUS_SUCCESS) entry.alive = false;
        return state.destroy_status;
    };
    sdk.backend_initialize = [](Native native) {
        auto& entry = object(native);
        callback();
        ++state.initializes;
        const auto kind = entry.snapshot.kind;
        entry.snapshot = {};
        entry.snapshot.kind = kind;
        return static_cast<cudnnStatus_t>(control.initialize_status.load());
    };
    sdk.backend_finalize = [](Native native) {
        auto& entry = object(native);
        callback();
        ++state.finalizes;
        const auto status = static_cast<cudnnStatus_t>(control.finalize_status.load());
        entry.snapshot.is_finalized = status == CUDNN_STATUS_SUCCESS;
        return status;
    };
    sdk.backend_set = [](Native native, cudnnBackendAttributeName_t attribute,
                         cudnnBackendAttributeType_t value_type, std::int64_t count,
                         const void* input) {
        auto& entry = object(native);
        callback();
        ++state.sets;
        const auto name = static_cast<std::uint32_t>(attribute);
        const auto type = static_cast<std::uint32_t>(value_type);
        state.last_name = name;
        state.last_type = type;
        state.last_count = count;
        state.last_has_array = input;
        if (state.capture_set_logging) {
            // This opt-in control models the audited scalar logging path,
            // which reads the original positive prefix before finalized checks.
            base::require(entry.snapshot.kind == 17 && name == 906 && type == 3 && input &&
                          count > 0 && count <= 65536);
            const auto* first = static_cast<const std::byte*>(input);
            state.logged_set_bytes.assign(first, first + static_cast<std::size_t>(count) * 8);
        }
        if (entry.snapshot.is_finalized || (!input && count > 0)) return CUDNN_STATUS_BAD_PARAM;
        if (entry.snapshot.kind == 17 && name == 909) {
            if (type != 20) return CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE;
            if (count != 1) return CUDNN_STATUS_BAD_PARAM;
        }
        if ((entry.snapshot.kind == 0 || entry.snapshot.kind == 13 ||
             (entry.snapshot.kind >= 10 && entry.snapshot.kind <= 12)) &&
            count != 1)
            return CUDNN_STATUS_BAD_PARAM;
        Attribute value{type, count, {}, {}};
        if (is_child(name)) {
            if (type != 15) return CUDNN_STATUS_BAD_PARAM;
            Native child{};
            if (input) std::memcpy(&child, input, sizeof(child));
            state.last_source = child;
            if (child) {
                const auto& original = object(child).snapshot;
                if (original.kind != child_kind(name)) return CUDNN_STATUS_BAD_PARAM;
                value.children.push_back(std::make_shared<const Snapshot>(original));
            } else {
                auto cleared = std::make_shared<Snapshot>();
                cleared->kind = child_kind(name);
                value.children.push_back(std::move(cleared));
            }
            value.count = 1;
        } else {
            if (count < 0 || count > 65536 || (!input && count)) return CUDNN_STATUS_BAD_PARAM;
            if (entry.snapshot.kind == 17 && (name == 902 || name == 903))
                value.count = std::min<std::int64_t>(count, 9);
            const auto size = name == 1503 && input
                                  ? std::size_t{8}
                                  : static_cast<std::size_t>(value.count) * width(type);
            if (size) {
                const auto* source = static_cast<const std::byte*>(input);
                value.bytes.assign(source, source + size);
            }
        }
        // Deliberately commit before returning an injected error. Tests must
        // distinguish dirty configuration from uncertain native ownership.
        entry.snapshot.attributes[name] = std::move(value);
        return static_cast<cudnnStatus_t>(control.set_status.load());
    };
    sdk.backend_get = [](Native native, cudnnBackendAttributeName_t attribute,
                         cudnnBackendAttributeType_t value_type, std::int64_t requested,
                         std::int64_t* count, void* output) {
        auto& entry = object(native);
        callback();
        ++state.gets;
        const auto name = static_cast<std::uint32_t>(attribute);
        const auto type = static_cast<std::uint32_t>(value_type);
        state.last_name = name;
        state.last_type = type;
        state.last_count = requested;
        state.last_has_array = output;
        state.last_has_count = count;
        if (!entry.snapshot.is_finalized || (!count && !output) || (output && requested < 0))
            return CUDNN_STATUS_BAD_PARAM;
        if (entry.snapshot.kind == 0 || entry.snapshot.kind == 13)
            return CUDNN_STATUS_NOT_SUPPORTED;
        if ((entry.snapshot.kind == 17 && name == 900) || control.fail_get_name.load() == name)
            return CUDNN_STATUS_NOT_SUPPORTED;
        if (entry.snapshot.kind == 17 && name == 909 && type != 3)
            return CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE;
        const auto found = entry.snapshot.attributes.find(name);
        const auto value = found == entry.snapshot.attributes.end()
                               ? default_attribute(entry.snapshot.kind, name)
                               : found->second;
        if (is_child(name)) {
            if (type != 15 || (output && requested != 1)) return CUDNN_STATUS_BAD_PARAM;
            if (output) {
                Native target{};
                std::memcpy(&target, output, sizeof(target));
                if (!target || value.children.size() != 1 || !value.children[0])
                    return CUDNN_STATUS_BAD_PARAM;
                auto& destination = object(target);
                if (destination.snapshot.kind != child_kind(name)) return CUDNN_STATUS_BAD_PARAM;
                state.last_destination = target;
                destination.snapshot = *value.children[0];
            }
            if (count) *count = 1;
        } else {
            if (type != value.type && !(entry.snapshot.kind == 17 && name == 909 && type == 3) &&
                !(name == 1503 && (type == 3 || type == 4 || type == 5 || type == 23)))
                return CUDNN_STATUS_BAD_PARAM;
            const bool is_array = (entry.snapshot.kind == 17 && (name == 902 || name == 903)) ||
                                  (entry.snapshot.kind == 1 && name >= 102 && name <= 105) ||
                                  (entry.snapshot.kind == 24 && name >= 1703 && name <= 1706);
            if (!is_array && output && requested != 1) return CUDNN_STATUS_BAD_PARAM;
            const auto copied = is_array ? (output ? std::min(requested, value.count) : value.count)
                                         : std::int64_t{1};
            const auto byte_count =
                name == 1503 && output ? std::size_t{8}
                : (name == 1524 || (entry.snapshot.kind == 17 && name == 909)) && output
                    ? std::size_t{4}
                    : static_cast<std::size_t>(copied) * width(type);
            if (output && byte_count) {
                base::require(byte_count <= value.bytes.size());
                std::memcpy(output, value.bytes.data(), byte_count);
                if (control.corrupt_bool.load() && type == 2) {
                    const std::uint8_t invalid = 2;
                    std::memcpy(output, &invalid, 1);
                }
                if (control.corrupt_enum.load() && byte_count >= 4) {
                    const std::uint32_t invalid = UINT32_MAX;
                    std::memcpy(output, &invalid, sizeof(invalid));
                }
            }
            if (count) *count = copied;
        }
        if (count && control.corrupt_count.load()) *count = INT64_MAX;
        return static_cast<cudnnStatus_t>(control.get_status.load());
    };
    if (missing == 1) sdk.version = nullptr;
    if (missing == 2) sdk.backend_create = nullptr;
    if (missing == 3) sdk.backend_destroy = nullptr;
    if (missing == 4) sdk.backend_initialize = nullptr;
    if (missing == 5) sdk.backend_finalize = nullptr;
    if (missing == 6) sdk.backend_set = nullptr;
    if (missing == 7) sdk.backend_get = nullptr;
    return sdk;
}
inline CudaSessionDispatch session_dispatch(unsigned int missing = 0) {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch(missing));
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
inline unsigned int alive() {
    unsigned int count{};
    for (const auto& value : state.objects) count += value.alive;
    return count;
}
}  // namespace glimmer::tests::dnn_backend_graph
