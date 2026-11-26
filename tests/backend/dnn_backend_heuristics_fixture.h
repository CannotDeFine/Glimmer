#pragma once

#include "dnn_backend_graph_fixture.h"
#include "dnn_convolution_fixture.h"

#include <set>

// Independent, deliberately small CPU model of generated planning metadata.
// Its scale operation is not a model of NVIDIA engine selection or GPU kernels.
namespace glimmer::tests::dnn_backend_heuristics {
namespace graph = glimmer::tests::dnn_backend_graph;
namespace driver = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
using namespace backend::cuda;
using Native = cudnnBackendDescriptor_t;
using Snapshot = graph::Snapshot;
using Attribute = graph::Attribute;
inline constexpr std::int64_t kWorkspaceBytes = 32;
struct State {
    CudaDnnDispatch original;
    std::set<Native> private_packs;
    std::vector<Native> visited, written;
    unsigned int executes{}, private_creates{}, private_destroys{}, planning_gets{};
    unsigned int count_stores{}, private_observations{};
    std::int64_t requested{}, raw_candidates{4};
    std::vector<unsigned int> accepted_candidates{0, 2};
    std::function<void()> callback;
    cudnnStatus_t execute_status{CUDNN_STATUS_SUCCESS};
    int fail_after{-1};
    bool issuing{}, lose_context{};
};
inline State state;
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
inline void reset() {
    graph::reset();
    driver::reset();
    state = {};
}
inline void callback() {
    graph::callback();
    if (state.callback) state.callback();
    if (state.lose_context) driver::driver.current = nullptr;
}
inline bool planning(std::uint32_t kind) {
    return kind == 2 || kind == 3 || kind == 4 || kind == 5 || kind == 7 || kind == 8 ||
           kind == 15 || kind == 16;
}
inline bool child_attribute(std::uint32_t name) {
    return name == 201 || name == 300 || name == 302 || name == 401 || name == 801 || name == 1300;
}
template <typename T>
inline T scalar_value(const Snapshot& object, std::uint32_t name) {
    const auto& bytes = object.attributes.at(name).bytes;
    require(bytes.size() >= sizeof(T), "complete independent native scalar");
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(value));
    return value;
}
inline void scalar(Snapshot& object, std::uint32_t name, std::uint32_t type,
                   std::vector<std::byte> bytes) {
    object.attributes[name] = {type, 1, std::move(bytes), {}};
}
template <std::uint32_t Kind>
inline Snapshot knob(unsigned int index) {
    Snapshot output{Kind, true, {}};
    if (Kind == 8) {
        scalar(output, 1200, 9, graph::scalar(index));
        scalar(output, 1201, 3, graph::scalar(std::int64_t{16} + index));
        scalar(output, 1202, 3, graph::scalar(std::int64_t{1} + index));
        scalar(output, 1203, 3, graph::scalar(std::int64_t{2}));
    } else {
        scalar(output, 600, 9, graph::scalar(index));
        scalar(output, 601, 3, graph::scalar(std::int64_t{31} + index));
    }
    return output;
}
inline Snapshot generated_config(const Snapshot& heuristic, unsigned int index) {
    Snapshot engine{2, true, {}};
    engine.attributes[1300] = heuristic.attributes.at(201);
    scalar(engine, 1301, 3, graph::scalar(std::int64_t{100} + index));
    Snapshot config{3, true, {}};
    config.attributes[300] = {15, 1, {}, {std::make_shared<const Snapshot>(engine)}};
    auto& choices = config.attributes[302];
    choices.type = 15;
    choices.count = 3;
    for (unsigned int i = 0; i != 3; ++i)
        choices.children.push_back(std::make_shared<const Snapshot>(knob<7>(i)));
    return config;
}
inline std::vector<std::shared_ptr<const Snapshot>> normalized_choices(const Snapshot& config) {
    std::map<std::uint32_t, std::shared_ptr<const Snapshot>> normalized;
    const auto found = config.attributes.find(302);
    if (found != config.attributes.end())
        for (const auto& child : found->second.children)
            normalized[scalar_value<std::uint32_t>(*child, 600)] = child;
    std::vector<std::shared_ptr<const Snapshot>> result;
    for (const auto& [key, value] : normalized) {
        (void)key;
        result.push_back(value);
    }
    return result;
}
inline void require_device(const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<CUdeviceptr>(pointer);
    for (const auto& [start, memory] : driver::driver.memory)
        if (address >= start && address - start <= memory.bytes.size() &&
            bytes <= memory.bytes.size() - (address - start)) {
            require(memory.context == driver::current(), "native memory uses its current owner");
            return;
        }
    throw std::runtime_error("independent Execute sees an unowned or short allocation");
}
inline CudaDnnDispatch dispatch() {
    state.original = graph::dispatch();
    auto sdk = state.original;
    sdk.create = driver::dispatch().create;
    sdk.backend_create = [](cudnnBackendDescriptorType_t kind, Native* output) {
        const auto status = state.original.backend_create(kind, output);
        if (status == CUDNN_STATUS_SUCCESS && kind == CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR &&
            state.issuing) {
            require(state.private_packs.insert(*output).second, "fresh command-private pack");
            ++state.private_creates;
        }
        return status;
    };
    sdk.backend_destroy = [](Native object) {
        const auto status = state.original.backend_destroy(object);
        if (state.private_packs.erase(object)) ++state.private_destroys;
        return status;
    };
    sdk.backend_set = [](Native native, cudnnBackendAttributeName_t attribute,
                         cudnnBackendAttributeType_t value_type, std::int64_t count,
                         const void* input) {
        auto& object = graph::object(native).snapshot;
        if (!planning(object.kind))
            return state.original.backend_set(native, attribute, value_type, count, input);
        callback();
        const auto name = static_cast<std::uint32_t>(attribute);
        const auto type = static_cast<std::uint32_t>(value_type);
        if (object.is_finalized || !input || count <= 0 || count > 100)
            return CUDNN_STATUS_BAD_PARAM;
        if (name == 302 && count > 43) return CUDNN_STATUS_BAD_PARAM;
        Attribute value{type, count, {}, {}};
        if (child_attribute(name)) {
            require(type == 15, "native copied-descriptor attribute type");
            const auto* children = static_cast<const Native*>(input);
            for (std::int64_t i = 0; i != count; ++i)
                value.children.push_back(
                    std::make_shared<const Snapshot>(graph::object(children[i]).snapshot));
            if (name == 302) {
                std::set<std::uint32_t> types;
                for (const auto& choice : value.children)
                    if (!types.insert(scalar_value<std::uint32_t>(*choice, 600)).second)
                        return CUDNN_STATUS_BAD_PARAM;
            }
        } else {
            const auto width = type == 0 || type == 6 ? sizeof(void*) : graph::width(type);
            const auto* first = static_cast<const std::byte*>(input);
            value.bytes.assign(first, first + static_cast<std::size_t>(count) * width);
            if (name == 800) {
                require(type == 0 && count == 1, "native graph handle contract");
                (void)base::handle(*static_cast<const cudnnHandle_t*>(input));
            }
        }
        if (name == 801) {
            auto& previous = object.attributes[name];
            previous.type = type;
            previous.count += count;
            previous.children.insert(previous.children.end(), value.children.begin(),
                                     value.children.end());
        } else {
            object.attributes[name] = std::move(value);
        }
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.backend_finalize = [](Native native) {
        const auto kind = graph::object(native).snapshot.kind;
        if (planning(kind) && kind != 7 && kind != 8 && kind != 16)
            require(driver::current(), "native planning uses a current context");
        return state.original.backend_finalize(native);
    };
    sdk.backend_get = [](Native native, cudnnBackendAttributeName_t attribute,
                         cudnnBackendAttributeType_t value_type, std::int64_t requested,
                         std::int64_t* count, void* output) {
        const auto& object = graph::object(native).snapshot;
        if (!planning(object.kind))
            return state.original.backend_get(native, attribute, value_type, requested, count,
                                              output);
        callback();
        ++state.planning_gets;
        state.requested = requested;
        state.visited.clear();
        state.written.clear();
        const auto name = static_cast<std::uint32_t>(attribute);
        const auto type = static_cast<std::uint32_t>(value_type);
        if (!object.is_finalized) return CUDNN_STATUS_NOT_INITIALIZED;
        if ((!count && !output) || (output && requested < 0)) return CUDNN_STATUS_BAD_PARAM;
        const auto publish_count = [count](std::int64_t value) {
            if (count) {
                *count = value;
                ++state.count_stores;
            }
        };
        if (name == 202 || name == 1302 || name == 302) {
            require(type == 15, "native generated-descriptor array ABI");
            std::vector<std::shared_ptr<const Snapshot>> generated;
            if (name == 202) {
                if (!output) {
                    ++state.private_observations;
                    publish_count(state.raw_candidates);
                    return CUDNN_STATUS_SUCCESS;
                }
                for (const auto index : state.accepted_candidates)
                    if (index < static_cast<unsigned int>(state.raw_candidates))
                        generated.push_back(
                            std::make_shared<const Snapshot>(generated_config(object, index)));
            } else if (name == 1302) {
                for (unsigned int i = 0; i != 3; ++i)
                    generated.push_back(std::make_shared<const Snapshot>(knob<8>(i)));
            } else {
                generated = normalized_choices(object);
            }
            const auto capacity = static_cast<std::int64_t>(generated.size());
            const auto written = name == 1302 && requested == 0
                                     ? capacity
                                     : std::clamp(requested, std::int64_t{0}, capacity);
            if (!output) {
                ++state.private_observations;
                publish_count(written);
                return CUDNN_STATUS_SUCCESS;
            }
            auto* destinations = static_cast<Native*>(output);
            if (name == 302) {
                // This native path validates even the unwritten requested tail.
                for (std::int64_t i = 0; i != requested; ++i) {
                    state.visited.push_back(destinations[i]);
                    if (!destinations[i]) return CUDNN_STATUS_BAD_PARAM;
                    if (graph::object(destinations[i]).snapshot.kind != 7)
                        return CUDNN_STATUS_BAD_PARAM;
                }
            }
            for (std::int64_t i = 0; i != written; ++i) {
                if (name != 302) state.visited.push_back(destinations[i]);
                if (!destinations[i]) return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
                auto& target = graph::object(destinations[i]).snapshot;
                if (target.kind != generated[i]->kind)
                    return CUDNN_STATUS_BAD_PARAM_DESCRIPTOR_TYPE;
                if (name == 202 && target.is_finalized) return CUDNN_STATUS_BAD_PARAM;
                target = *generated[i];
                state.written.push_back(destinations[i]);
                if (state.fail_after == i) return CUDNN_STATUS_BAD_PARAM;
            }
            publish_count(written);
            return CUDNN_STATUS_SUCCESS;
        }
        if (output && requested != 1) return CUDNN_STATUS_BAD_PARAM;
        if (name == 300 || name == 401 || name == 1300) {
            if (output) {
                Native destination{};
                std::memcpy(&destination, output, sizeof(destination));
                const auto& source = *object.attributes.at(name).children.front();
                auto& target = graph::object(destination).snapshot;
                require(target.kind == source.kind, "original single-copy destination kind");
                target = source;
                state.written.push_back(destination);
            }
        } else {
            const auto data = name == 402   ? graph::scalar(kWorkspaceBytes)
                              : name == 802 ? graph::scalar(std::int64_t{7})
                                            : object.attributes.at(name).bytes;
            if (output) std::memcpy(output, data.data(), graph::width(type));
        }
        publish_count(1);
        return CUDNN_STATUS_SUCCESS;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact original SDK ABI.
    sdk.backend_execute = [](cudnnHandle_t handle, Native plan, Native pack) {
        (void)base::handle(handle);
        require(graph::object(plan).snapshot.kind == 5 &&
                    graph::object(plan).snapshot.is_finalized && state.private_packs.contains(pack),
                "original Execute receives finalized generated plan and private pack");
        const auto& values = graph::object(pack).snapshot.attributes;
        const auto& ids = values.at(1000).bytes;
        const auto& pointers = values.at(1001).bytes;
        require(ids.size() == 3 * sizeof(std::int64_t) && pointers.size() == 3 * sizeof(void*),
                "independent Execute receives all three tensor bindings");
        std::array<void*, 3> buffers{};
        for (std::size_t i = 0; i != buffers.size(); ++i) {
            std::int64_t uid{};
            std::memcpy(&uid, ids.data() + i * sizeof(uid), sizeof(uid));
            require(uid >= 11 && uid <= 13, "known independently configured tensor UID");
            std::memcpy(&buffers[static_cast<std::size_t>(uid - 11)],
                        pointers.data() + i * sizeof(void*), sizeof(void*));
        }
        require_device(buffers[0], 16);
        require_device(buffers[1], 16);
        float scale{};
        require(buffers[2], "captured non-null by-value coefficient");
        std::memcpy(&scale, buffers[2], sizeof(scale));
        void* workspace{};
        std::memcpy(&workspace, values.at(1003).bytes.data(), sizeof(workspace));
        require_device(workspace, kWorkspaceBytes);
        const auto* input = static_cast<const float*>(buffers[0]);
        auto* output = static_cast<float*>(buffers[1]);
        for (std::size_t i = 0; i != 4; ++i) output[i] = scale * input[i];
        static_cast<std::byte*>(workspace)[0] = std::byte{0x5a};
        ++state.executes;
        return state.execute_status;
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = driver::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_backend_heuristics
