#include "cuda_native_graph_effects.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

namespace glimmer::interceptor {
namespace {
struct GraphEffectsReader {
    const CudaNativeGraphMetadataDispatch& dispatch;
    const CudaNativeGraphEffectsLimits limits;
    std::size_t node_count{};
    std::vector<CUgraph> path;
    std::vector<CudaNativeGraphEventEffect> effects;
    std::vector<std::pair<CUgraphNode, CUgraphNodeType>> known_nodes;
    std::vector<CUgraphNode> scopes;
    struct Child {
        CUgraphNode node{};
        std::size_t node_begin{}, node_end{}, event_begin{}, event_end{};
        std::vector<CUgraphNode> ancestors;
    };
    std::vector<Child> children;
    std::vector<CUgraphNode> ancestors;

    CUresult read(CUgraph graph) {
        if (path.size() >= limits.depth) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        if (std::ranges::find(path, graph) != path.end()) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        path.push_back(graph);
        const auto result = read_nodes(graph);
        path.pop_back();
        return result;
    }
    CUresult read_nodes(CUgraph graph) {
        std::size_t count = std::numeric_limits<std::size_t>::max();
        auto result = dispatch.nodes(graph, nullptr, &count);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        if (count == std::numeric_limits<std::size_t>::max()) {
            return CUDA_ERROR_UNKNOWN;
        }
        if (count > limits.nodes - node_count) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        std::vector<CUgraphNode> nodes(count);
        if (count != 0) {
            auto filled = count;
            result = dispatch.nodes(graph, nodes.data(), &filled);
            if (result != CUDA_SUCCESS) {
                return result;
            }
            if (filled != count) {
                return CUDA_ERROR_NOT_READY;
            }
        }
        std::size_t verified = std::numeric_limits<std::size_t>::max();
        result = dispatch.nodes(graph, nullptr, &verified);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        if (verified != count) {
            return CUDA_ERROR_NOT_READY;
        }
        node_count += count;
        for (auto current = nodes.begin(); current != nodes.end(); ++current) {
            if (!*current || std::find(nodes.begin(), current, *current) != current) {
                return CUDA_ERROR_UNKNOWN;
            }
            // 15 is representable by this enum's range in supported SDKs, but
            // is not a supported node type. Do not use an out-of-range sentinel.
            auto type = static_cast<CUgraphNodeType>(15);
            result = dispatch.node_type(*current, &type);
            if (result != CUDA_SUCCESS) {
                return result;
            }
            if (type == static_cast<CUgraphNodeType>(15)) {
                return CUDA_ERROR_UNKNOWN;
            }
            known_nodes.emplace_back(*current, type);
            scopes.push_back(ancestors.empty() ? nullptr : ancestors.back());
            result = read_node(*current, type);
            if (result != CUDA_SUCCESS) {
                return result;
            }
        }
        return CUDA_SUCCESS;
    }
    CUresult read_node(CUgraphNode node, CUgraphNodeType type) {
        switch (type) {
            case CU_GRAPH_NODE_TYPE_KERNEL:
            case CU_GRAPH_NODE_TYPE_MEMCPY:
            case CU_GRAPH_NODE_TYPE_MEMSET:
            case CU_GRAPH_NODE_TYPE_EMPTY:
                return CUDA_SUCCESS;
            case CU_GRAPH_NODE_TYPE_GRAPH: {
                if (!dispatch.child_graph) {
                    return CUDA_ERROR_NOT_SUPPORTED;
                }
                CUgraph child = nullptr;
                const auto result = dispatch.child_graph(node, &child);
                if (result != CUDA_SUCCESS || !child)
                    return result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result;
                const auto node_begin = known_nodes.size();
                const auto event_begin = effects.size();
                auto parents = ancestors;
                ancestors.push_back(node);
                const auto discovered = read(child);
                ancestors.pop_back();
                if (discovered != CUDA_SUCCESS) return discovered;
                children.push_back({node, node_begin, known_nodes.size(), event_begin,
                                    effects.size(), std::move(parents)});
                return CUDA_SUCCESS;
            }
            case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
            case CU_GRAPH_NODE_TYPE_WAIT_EVENT: {
                const bool is_record = type == CU_GRAPH_NODE_TYPE_EVENT_RECORD;
                const auto getter = is_record ? dispatch.recorded_event : dispatch.waited_event;
                if (!getter) {
                    return CUDA_ERROR_NOT_SUPPORTED;
                }
                CUevent event = nullptr;
                const auto result = getter(node, &event);
                if (result != CUDA_SUCCESS || !event) {
                    return result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result;
                }
                effects.push_back({node, event,
                                   is_record ? CudaNativeGraphEventKind::kRecord
                                             : CudaNativeGraphEventKind::kWait});
                return CUDA_SUCCESS;
            }
            default:
                return CUDA_ERROR_NOT_SUPPORTED;
        }
    }
};
}  // namespace

CUresult CudaNativeGraphEffects::replace_event(CUgraphNode node, CUevent event,
                                               CudaNativeGraphEventKind kind,
                                               CudaNativeGraphEffects& output) const noexcept {
    if (!data_ || !node || !event || output) return CUDA_ERROR_INVALID_VALUE;
    const auto matches = [&](const auto& effect) {
        return effect.node == node && effect.kind == kind;
    };
    if (std::ranges::count_if(data_->events, matches) != 1) return CUDA_ERROR_INVALID_HANDLE;
    try {
        auto replacement = std::make_shared<Data>(*data_);
        std::ranges::find_if(replacement->events, matches)->event = event;
        output.data_ = std::move(replacement);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaNativeGraphEffects::preserve_node(CUgraphNode node,
                                               std::optional<CUgraphNodeType> expected_type,
                                               CudaNativeGraphEffects& output) const noexcept {
    if (!data_ || !node || output) return CUDA_ERROR_INVALID_VALUE;
    const auto matches = [&](const auto& value) { return value.first == node; };
    if (std::ranges::count_if(data_->nodes, matches) != 1) return CUDA_ERROR_INVALID_HANDLE;
    const auto type = std::ranges::find_if(data_->nodes, matches)->second;
    if ((expected_type && *expected_type != type) ||
        (type != CU_GRAPH_NODE_TYPE_KERNEL && type != CU_GRAPH_NODE_TYPE_MEMCPY &&
         type != CU_GRAPH_NODE_TYPE_MEMSET))
        return CUDA_ERROR_NOT_SUPPORTED;
    output.data_ = data_;
    return CUDA_SUCCESS;
}

std::vector<CUgraphNode> CudaNativeGraphEffects::replacement_aliases(
    const Data& replacement, CUgraphNode original_scope) const {
    std::vector<CUgraphNode> aliases(replacement.nodes.size());
    struct ScopeMatch {
        CUgraphNode original{};
        CUgraphNode replacement{};
    };
    const auto map_scope = [&](auto&& self, ScopeMatch scope) -> void {
        for (std::size_t index = 0; index < replacement.nodes.size(); ++index) {
            if (replacement.scopes[index] != scope.replacement) continue;
            const auto type = replacement.nodes[index].second;
            std::size_t old_count{}, new_count{}, candidate{};
            for (std::size_t other = 0; other < data_->nodes.size(); ++other) {
                if (data_->scopes[other] == scope.original && data_->nodes[other].second == type) {
                    ++old_count;
                    candidate = other;
                }
            }
            for (std::size_t other = 0; other < replacement.nodes.size(); ++other)
                new_count += replacement.scopes[other] == scope.replacement &&
                             replacement.nodes[other].second == type;
            if (old_count != 1 || new_count != 1) continue;
            aliases[index] = data_->nodes[candidate].first;
            if (type == CU_GRAPH_NODE_TYPE_GRAPH)
                self(self, {aliases[index], replacement.nodes[index].first});
        }
    };
    map_scope(map_scope, {original_scope, nullptr});
    return aliases;
}

void CudaNativeGraphEffects::rebind_children(Data& output, const Data& replacement,
                                             const std::vector<CUgraphNode>& aliases,
                                             std::size_t event_offset) const {
    const auto alias = [&](CUgraphNode node) {
        CUgraphNode value{};
        std::size_t matches{};
        for (std::size_t index = 0; index < replacement.nodes.size(); ++index) {
            if (replacement.nodes[index].first == node) {
                value = aliases[index];
                ++matches;
            }
        }
        return matches == 1 ? value : nullptr;
    };
    for (const auto& child : replacement.children) {
        const auto origin = alias(child.node);
        const auto old = std::ranges::find_if(
            data_->children, [&](const auto& value) { return origin && value.node == origin; });
        Child rebound{origin,
                      old == data_->children.end() ? 0 : old->node_begin,
                      old == data_->children.end() ? 0 : old->node_end,
                      event_offset + child.event_begin,
                      event_offset + child.event_end,
                      {}};
        rebound.ancestors.reserve(child.ancestors.size());
        for (const auto parent : child.ancestors) rebound.ancestors.push_back(alias(parent));
        output.children.push_back(std::move(rebound));
    }
    for (std::size_t index = 0; index < replacement.events.size(); ++index)
        output.events[event_offset + index].node = alias(replacement.events[index].node);
}

CUresult CudaNativeGraphEffects::replace_graph(const CudaNativeGraphEffects& replacement,
                                               CudaNativeGraphEffects& output) const noexcept {
    if (!data_ || !replacement || output) return CUDA_ERROR_INVALID_VALUE;
    if (data_->node_count != replacement.node_count()) return CUDA_ERROR_NOT_SUPPORTED;
    try {
        auto replaced = std::make_shared<Data>(*data_);
        replaced->events = replacement.data_->events;
        replaced->children.clear();
        const auto aliases = replacement_aliases(*replacement.data_, nullptr);
        rebind_children(*replaced, *replacement.data_, aliases, 0);
        output.data_ = std::move(replaced);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult CudaNativeGraphEffects::replace_child(CUgraphNode node,
                                               const CudaNativeGraphEffects& replacement,
                                               CudaNativeGraphEffects& output) const noexcept {
    if (!data_ || !node || !replacement || output) return CUDA_ERROR_INVALID_VALUE;
    const auto matches = [&](const auto& child) { return child.node == node; };
    if (std::ranges::count_if(data_->children, matches) != 1) return CUDA_ERROR_INVALID_HANDLE;
    const auto& child = *std::ranges::find_if(data_->children, matches);
    if (child.node_end - child.node_begin != replacement.node_count() ||
        child.event_end - child.event_begin != replacement.events().size())
        return CUDA_ERROR_NOT_SUPPORTED;
    try {
        auto replaced = std::make_shared<Data>(*data_);
        std::copy(replacement.data_->events.begin(), replacement.data_->events.end(),
                  replaced->events.begin() + static_cast<std::ptrdiff_t>(child.event_begin));
        std::erase_if(replaced->children, [&](const auto& value) {
            return std::ranges::find(value.ancestors, node) != value.ancestors.end();
        });
        const auto aliases = replacement_aliases(*replacement.data_, node);
        const auto old_size = replaced->children.size();
        rebind_children(*replaced, *replacement.data_, aliases, child.event_begin);
        for (std::size_t index = old_size; index < replaced->children.size(); ++index) {
            auto& parents = replaced->children[index].ancestors;
            parents.insert(parents.begin(), node);
            parents.insert(parents.begin(), child.ancestors.begin(), child.ancestors.end());
        }
        output.data_ = std::move(replaced);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult prepare_native_graph_effects(CUgraph graph,
                                      const CudaNativeGraphMetadataDispatch& dispatch,
                                      CudaNativeGraphEffects& output,
                                      CudaNativeGraphEffectsLimits limits) noexcept {
    if (!graph || output || limits.nodes == 0 || limits.nodes > 4096 || limits.depth == 0 ||
        limits.depth > 32) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!dispatch.nodes || !dispatch.node_type) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    try {
        GraphEffectsReader reader{dispatch, limits, 0, {}, {}, {}, {}, {}, {}};
        const auto result = reader.read(graph);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        auto data = std::make_shared<CudaNativeGraphEffects::Data>();
        data->node_count = reader.node_count;
        data->events = std::move(reader.effects);
        data->nodes = std::move(reader.known_nodes);
        data->scopes = std::move(reader.scopes);
        data->children.reserve(reader.children.size());
        for (auto& child : reader.children) {
            data->children.push_back({child.node, child.node_begin, child.node_end,
                                      child.event_begin, child.event_end,
                                      std::move(child.ancestors)});
        }
        output.data_ = std::move(data);
        return CUDA_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
}  // namespace glimmer::interceptor
