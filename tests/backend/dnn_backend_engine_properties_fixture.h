#pragma once

#include "dnn_convolution_fixture.h"
#include "internal/cuda_execution_session.h"

#include <functional>
#include <map>
#include <string>

namespace glimmer::tests::dnn_backend_engine_properties {
using namespace backend::cuda;
namespace driver = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
using Native = cudnnBackendDescriptor_t;

// Independent host-only model of the audited 9.10.2 row contracts. The data
// below is deliberately synthetic: it is not a vendor engine or JSON parser.
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename T>
std::vector<std::byte> bytes(std::span<const T> values) {
    const auto view = std::as_bytes(values);
    return {view.begin(), view.end()};
}
template <typename T>
std::vector<std::byte> scalar(T value) {
    return bytes(std::span<const T>{&value, 1});
}
struct Header {
    bool initialized{true}, finalized{};
};
struct Layout {
    Header header;
    // Deliberate poison, NOT a claim about an SDK constructor's default bytes.
    std::int64_t uid{-7001};
    std::vector<std::uint32_t> types{3};
};
struct Property {
    Header header;
    std::int32_t id{};
    bool populated{}, json_mode{};
    std::array<std::byte, 456> copied_value{};
    std::string json;
};
struct Object {
    bool alive{};
    unsigned int destroy_attempts{};
    std::uint32_t kind{};
    Header header;
    Layout layout;
    Property property;
    std::optional<Property> child_property;
    std::vector<Layout> layouts;
    std::vector<std::uint32_t> numerical, behavior;
};
struct State {
    struct GetCall {
        std::uint32_t name;
        std::int64_t requested;
        bool has_count, has_array;
    };
    std::array<Object, 128> objects{};
    unsigned int creates{}, destroys{}, initializes{}, finalizes{}, sets{}, gets{}, versions{};
    unsigned int discoveries{}, handle_copies{};
    std::map<std::uint32_t, unsigned int> get_calls;
    std::vector<GetCall> get_history;
    std::vector<Native> validated, written;
    std::function<void()> callback;
    std::vector<std::uint32_t> numerical{1, 4, 9}, behavior{0, 3};
    std::vector<Layout> layouts{
        {{true, true}, 101, {0, 2}}, {{true, true}, 202, {1}}, {{true, true}, 303, {0, 1, 3}}};
    cudnnStatus_t set_status{CUDNN_STATUS_SUCCESS}, finalize_status{CUDNN_STATUS_SUCCESS};
    std::optional<std::uint32_t> failed_get, corrupt_count;
    unsigned int fail_after_copy{};
    bool corrupt_enum{};
    std::int64_t last_requested{};
    std::uint32_t last_name{};
};
inline State state;
inline void reset() {
    driver::reset();
    state = {};
}
inline Object& object(Native value) {
    base::owner();
    for (auto& item : state.objects)
        if (static_cast<void*>(&item) == value) {
            require(item.alive, "native descriptor remains alive");
            return item;
        }
    throw std::runtime_error("unknown independent engine/property descriptor");
}
inline void callback() {
    base::owner();
    if (state.callback) state.callback();
}
inline Header& header(Object& item) {
    if (item.kind == 9) return item.layout.header;
    if (item.kind == 38) return item.property.header;
    return item.header;
}
inline bool property_row(std::uint32_t kind, std::uint32_t name) {
    return (kind == 4 && name == 204) || (kind == 2 && name == 1307) || (kind == 5 && name == 407);
}
inline cudnnStatus_t status(unsigned int value) {
    return static_cast<cudnnStatus_t>(value);
}
inline cudnnStatus_t create(cudnnBackendDescriptorType_t kind, Native* output) {
    ++state.creates;
    callback();
    for (auto& item : state.objects)
        if (!item.alive) {
            item = {};
            item.alive = true;
            item.kind = static_cast<std::uint32_t>(kind);
            *output = &item;
            return CUDNN_STATUS_SUCCESS;
        }
    throw std::runtime_error("independent descriptor capacity");
}
inline cudnnStatus_t destroy(Native value) {
    auto& item = object(value);
    ++state.destroys;
    callback();
    require(++item.destroy_attempts == 1, "exactly one native destruction per object lifetime");
    item.alive = false;
    return CUDNN_STATUS_SUCCESS;
}
inline cudnnStatus_t initialize(Native value) {
    auto& item = object(value);
    ++state.initializes;
    callback();
    if (item.kind == 9) {
        // Native Initialize changes the header, not the opaque UID/bitmap.
        item.layout.header = {};
    } else if (item.kind == 38) {
        item.property = {};  // Full value reset, including ID=0 and JSON storage.
    } else {
        item.header = {};
        item.child_property.reset();
        item.layouts.clear();
        item.numerical.clear();
        item.behavior.clear();
    }
    return CUDNN_STATUS_SUCCESS;
}
inline cudnnStatus_t finalize(Native value) {
    auto& item = object(value);
    ++state.finalizes;
    callback();
    if (state.finalize_status != CUDNN_STATUS_SUCCESS) return state.finalize_status;
    if (item.kind == 38 && !item.property.populated && !item.property.json_mode) {
        require(driver::current(), "device discovery requires an actual live owner context");
        ++state.discoveries;
        item.property.populated = true;
        item.property.copied_value.fill(std::byte{0x51});
    }
    if (item.kind == 2) {
        item.numerical = state.numerical;
        item.behavior = state.behavior;
        item.layouts = state.layouts;
    }
    header(item).finalized = true;
    return CUDNN_STATUS_SUCCESS;
}
inline cudnnStatus_t set(Native value, cudnnBackendAttributeName_t raw_name,
                         cudnnBackendAttributeType_t raw_type, std::int64_t count,
                         const void* input) {
    auto& item = object(value);
    const auto name = static_cast<std::uint32_t>(raw_name);
    const auto type = static_cast<std::uint32_t>(raw_type);
    ++state.sets;
    callback();
    if (header(item).finalized) return CUDNN_STATUS_BAD_PARAM;
    if (item.kind == 9) return CUDNN_STATUS_NOT_SUPPORTED;
    if (property_row(item.kind, name)) {
        if (type != 15) return status(2010);
        if (count != 1) return CUDNN_STATUS_NOT_SUPPORTED;
        if (!input || !*static_cast<const Native*>(input)) return CUDNN_STATUS_BAD_PARAM;
        const auto& source = object(*static_cast<const Native*>(input));
        if (source.kind != 38) return CUDNN_STATUS_BAD_PARAM;
        item.child_property = source.property;
        return state.set_status;
    }
    if (item.kind != 38) return CUDNN_STATUS_NOT_SUPPORTED;
    auto& property = item.property;
    if (name == 2700) {
        if (type != 23) return status(2010);
        if (count != 1 || property.json_mode || property.populated || !input)
            return CUDNN_STATUS_BAD_PARAM;
        std::memcpy(&property.id, input, sizeof(property.id));
    } else if (name == 2701) {
        if (type != 0) return status(2010);
        if (count != 1 || property.json_mode || property.id != 0 || !input)
            return CUDNN_STATUS_BAD_PARAM;
        const auto handle = *static_cast<cudnnHandle_t const*>(input);
        if (!handle) return status(2002);
        (void)base::handle(handle);
        ++state.handle_copies;
        property.copied_value.fill(std::byte{0x6d});
        property.populated = true;
    } else if (name == 2702) {
        require(count >= 0, "unsafe negative JSON length never reaches native model");
        if (property.populated || property.id != 0) return CUDNN_STATUS_BAD_PARAM;
        require(count == 0 || input, "positive JSON source is materialized");
        if (count)
            property.json.assign(static_cast<const char*>(input), static_cast<std::size_t>(count));
        else
            property.json.clear();
        property.json_mode = true;
    } else {
        return CUDNN_STATUS_BAD_PARAM;
    }
    // Deliberately commit before an injected original error: no rollback fiction.
    return state.set_status;
}
inline cudnnStatus_t get(Native value, cudnnBackendAttributeName_t raw_name,
                         cudnnBackendAttributeType_t raw_type, std::int64_t requested,
                         std::int64_t* count, void* output) {
    auto& item = object(value);
    const auto name = static_cast<std::uint32_t>(raw_name);
    const auto type = static_cast<std::uint32_t>(raw_type);
    ++state.gets;
    ++state.get_calls[name];
    state.get_history.push_back({name, requested, count != nullptr, output != nullptr});
    state.last_requested = requested;
    state.last_name = name;
    callback();
    if (!header(item).finalized) return CUDNN_STATUS_NOT_INITIALIZED;
    if ((!count && !output) || (output && requested < 0)) return CUDNN_STATUS_BAD_PARAM;
    if (state.failed_get == name) return CUDNN_STATUS_INTERNAL_ERROR;
    const auto publish_count = [&](std::int64_t actual) {
        if (count) *count = state.corrupt_count == name ? INT64_MAX : actual;
    };
    const auto scalar_output = [&](const auto& actual) -> cudnnStatus_t {
        if (output && requested != 1) return CUDNN_STATUS_BAD_PARAM;
        if (output) std::memcpy(output, &actual, sizeof(actual));
        publish_count(1);
        return CUDNN_STATUS_SUCCESS;
    };
    const auto array_output = [&](const std::vector<std::uint32_t>& values) {
        const auto capacity = static_cast<std::int64_t>(values.size());
        const auto emitted = requested < 0    ? 0
                             : requested == 0 ? capacity
                                              : std::min(requested, capacity);
        if (output && emitted) {
            std::memcpy(output, values.data(), static_cast<std::size_t>(emitted) * 4);
            if (state.corrupt_enum) *static_cast<std::uint32_t*>(output) = UINT32_MAX;
        }
        publish_count(emitted);
        return CUDNN_STATUS_SUCCESS;
    };
    if (item.kind == 2 && name == 1303)
        return type == 11 ? array_output(item.numerical) : status(2010);
    if (item.kind == 2 && name == 1305)
        return type == 19 ? array_output(item.behavior) : status(2010);
    if (item.kind == 9 && name == 1100)
        return type == 3 ? scalar_output(item.layout.uid) : status(2010);
    if (item.kind == 9 && name == 1101)
        return type == 12 ? array_output(item.layout.types) : status(2010);
    if (item.kind == 2 && name == 1304) {
        if (type != 15) return status(2010);
        const auto emitted =
            requested <= 0 ? 0
                           : std::min(requested, static_cast<std::int64_t>(item.layouts.size()));
        state.validated.clear();
        state.written.clear();
        if (output) {
            auto* destinations = static_cast<Native*>(output);
            for (std::int64_t i = 0; i < requested; ++i) {
                if (!destinations[i] || object(destinations[i]).kind != 9)
                    return CUDNN_STATUS_BAD_PARAM;
                state.validated.push_back(destinations[i]);
            }
            for (std::int64_t i = 0; i < emitted; ++i) {
                object(destinations[i]).layout = item.layouts[static_cast<std::size_t>(i)];
                state.written.push_back(destinations[i]);
                if (state.fail_after_copy == state.written.size()) return CUDNN_STATUS_BAD_PARAM;
            }
        }
        publish_count(emitted);
        return CUDNN_STATUS_SUCCESS;
    }
    if (property_row(item.kind, name)) {
        if (type != 15) return status(2010);
        if (output) {
            if (requested != 1) return CUDNN_STATUS_NOT_SUPPORTED;
            auto destination = *static_cast<Native*>(output);
            if (!destination || object(destination).kind != 38) return CUDNN_STATUS_BAD_PARAM;
            if (!item.child_property) return status(2002);
            object(destination).property = *item.child_property;
            if (state.fail_after_copy) return CUDNN_STATUS_BAD_PARAM;
        }
        publish_count(1);
        return CUDNN_STATUS_SUCCESS;
    }
    if (item.kind == 38 && name == 2700)
        return type == 23 ? scalar_output(item.property.id) : status(2010);
    if (item.kind == 38 && name == 2701) return CUDNN_STATUS_NOT_SUPPORTED;
    if (item.kind == 38 && name == 2702) {
        auto& json = item.property.json;
        if (json.empty()) json = "{\"synthetic_device\":" + std::to_string(item.property.id) + "}";
        const auto size = static_cast<std::int64_t>(json.size());
        if (requested > 0 && requested < size) return status(2006);
        require(count, "JSON success requires caller count, not a fabricated optional pointer");
        if (requested > 0 && size) {
            require(output, "JSON successful byte copy requires actual caller array presence");
            std::memcpy(output, json.data(), json.size());
        }
        publish_count(size);
        return CUDNN_STATUS_SUCCESS;
    }
    // Unrelated legacy capacity observations, not new-row array oracles.
    if ((item.kind == 2 && name == 1302) || (item.kind == 4 && name == 202)) {
        publish_count(0);
        return CUDNN_STATUS_SUCCESS;
    }
    return CUDNN_STATUS_NOT_SUPPORTED;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = base::dispatch();
    sdk.version = [] {
        ++state.versions;
        callback();
        return std::size_t{91002};
    };
    sdk.backend_create = create;
    sdk.backend_destroy = destroy;
    sdk.backend_initialize = initialize;
    sdk.backend_finalize = finalize;
    sdk.backend_set = set;
    sdk.backend_get = get;
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = driver::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_backend_engine_properties
