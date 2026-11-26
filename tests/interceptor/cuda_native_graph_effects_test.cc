#include "internal/cuda_native_graph_effects.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {
using namespace glimmer::interceptor;
const auto kEvent = reinterpret_cast<CUevent>(0x1000);
const auto kOtherEvent = reinterpret_cast<CUevent>(0x2000);
constexpr CUresult kMetadataError = CUDA_ERROR_INVALID_CONTEXT;
void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}
enum class Fault {
    kNone,
    kCount,
    kFill,
    kVerify,
    kType,
    kChild,
    kRecord,
    kWait,
    kNullNode,
    kDuplicateNode,
    kUnwrittenCount,
    kUnwrittenType,
    kUnwrittenEvent,
    kNullChild,
    kGrowing,
    kShrinking,
    kThrow,
    kAllocation
};
struct Fake {
    struct Graph {
        std::vector<CUgraphNode> nodes;
        unsigned int queries{};
    };
    struct Node {
        CUgraphNodeType type{};
        CUevent event{};
        CUgraph child{};
    };
    static inline thread_local Fake* active{};
    std::vector<std::unique_ptr<Graph>> graphs;
    std::vector<std::unique_ptr<Node>> nodes;
    Fault fault{};
    unsigned int calls{};
    Fake() {
        require(!active, "one fake metadata owner per source caller");
        active = this;
    }
    ~Fake() {
        active = nullptr;
    }
    Fake(const Fake&) = delete;
    Fake& operator=(const Fake&) = delete;
    Fake(Fake&&) = delete;
    Fake& operator=(Fake&&) = delete;
    CUgraph graph() {
        graphs.push_back(std::make_unique<Graph>());
        return reinterpret_cast<CUgraph>(graphs.back().get());
    }
    CUgraphNode node(CUgraph graph, CUgraphNodeType type, CUevent event = nullptr,
                     CUgraph child = nullptr) {
        nodes.push_back(std::make_unique<Node>(Node{type, event, child}));
        const auto handle = reinterpret_cast<CUgraphNode>(nodes.back().get());
        reinterpret_cast<Graph*>(graph)->nodes.push_back(handle);
        return handle;
    }
    static CUresult get_nodes(CUgraph graph, CUgraphNode* output, std::size_t* count) {
        auto& fake = *active;
        ++fake.calls;
        if (fake.fault == Fault::kThrow) {
            throw std::runtime_error("metadata exception");
        }
        if (fake.fault == Fault::kAllocation) {
            throw std::bad_alloc();
        }
        auto& data = *reinterpret_cast<Graph*>(graph);
        if (!output) {
            ++data.queries;
            if (fake.fault == Fault::kCount ||
                (fake.fault == Fault::kVerify && data.queries == 2)) {
                return kMetadataError;
            }
            if (fake.fault != Fault::kUnwrittenCount) {
                *count =
                    data.nodes.size() +
                    static_cast<std::size_t>(fake.fault == Fault::kGrowing && data.queries == 2);
            }
            return CUDA_SUCCESS;
        }
        if (fake.fault == Fault::kFill) {
            return kMetadataError;
        }
        std::copy_n(data.nodes.begin(), std::min(*count, data.nodes.size()), output);
        *count = data.nodes.size();
        if (fake.fault == Fault::kShrinking) {
            --*count;
        }
        if (fake.fault == Fault::kNullNode) {
            output[0] = nullptr;
        }
        if (fake.fault == Fault::kDuplicateNode) {
            output[1] = output[0];
        }
        return CUDA_SUCCESS;
    }
    static CUresult get_type(CUgraphNode node, CUgraphNodeType* output) {
        ++active->calls;
        if (active->fault == Fault::kType) {
            return kMetadataError;
        }
        if (active->fault != Fault::kUnwrittenType) {
            *output = reinterpret_cast<Node*>(node)->type;
        }
        return CUDA_SUCCESS;
    }
    static CUresult get_child(CUgraphNode node, CUgraph* output) {
        ++active->calls;
        if (active->fault == Fault::kChild) {
            return kMetadataError;
        }
        if (active->fault != Fault::kNullChild) {
            *output = reinterpret_cast<Node*>(node)->child;
        }
        return CUDA_SUCCESS;
    }
    static CUresult get_record(CUgraphNode node, CUevent* output) {
        ++active->calls;
        if (active->fault == Fault::kRecord) {
            return kMetadataError;
        }
        if (active->fault != Fault::kUnwrittenEvent) {
            *output = reinterpret_cast<Node*>(node)->event;
        }
        return CUDA_SUCCESS;
    }
    static CUresult get_wait(CUgraphNode node, CUevent* output) {
        ++active->calls;
        if (active->fault == Fault::kWait) {
            return kMetadataError;
        }
        *output = reinterpret_cast<Node*>(node)->event;
        return CUDA_SUCCESS;
    }
    static CudaNativeGraphMetadataDispatch dispatch() {
        return {get_nodes, get_type, get_child, get_record, get_wait};
    }
};
void require_empty(const CudaNativeGraphEffects& output) {
    require(!output && output.node_count() == 0 && output.events().empty(),
            "failure must never publish partial graph effects");
}
void identities_and_lifetime() {
    CudaNativeGraphEffects effects;
    CUgraphNode recorded{};
    CUgraphNode waited{};
    {
        Fake fake;
        const auto root = fake.graph();
        const auto child = fake.graph();
        fake.node(root, CU_GRAPH_NODE_TYPE_KERNEL);
        fake.node(root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, child);
        recorded = fake.node(child, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
        waited = fake.node(child, CU_GRAPH_NODE_TYPE_WAIT_EVENT, kOtherEvent);
        fake.node(child, CU_GRAPH_NODE_TYPE_MEMCPY);
        fake.node(child, CU_GRAPH_NODE_TYPE_MEMSET);
        fake.node(child, CU_GRAPH_NODE_TYPE_EMPTY);
        require(prepare_native_graph_effects(root, Fake::dispatch(), effects) == CUDA_SUCCESS &&
                    effects && effects.node_count() == 7 && effects.events().size() == 2,
                "snapshot all original child-graph effects before instantiation");
        const auto before = fake.calls;
        require(prepare_native_graph_effects(root, Fake::dispatch(), effects) ==
                        CUDA_ERROR_INVALID_VALUE &&
                    fake.calls == before,
                "occupied output cannot discard a previous executable snapshot");
    }
    require(effects.events()[0].node == recorded && effects.events()[0].event == kEvent &&
                effects.events()[0].kind == CudaNativeGraphEventKind::kRecord &&
                effects.events()[1].node == waited && effects.events()[1].event == kOtherEvent &&
                effects.events()[1].kind == CudaNativeGraphEventKind::kWait,
            "owned CPU identities survive source destruction without dereferencing freed nodes");
    const auto copy = effects;
    effects = {};
    require(copy && copy.events().size() == 2 && copy.node_count() == 7,
            "snapshot copies own data, not source vector views");
    auto moved_source = copy;
    const auto moved = std::move(moved_source);
    // NOLINTNEXTLINE(bugprone-use-after-move): The snapshot promises an empty moved-from owner.
    require_empty(moved_source);
    require(moved && moved.node_count() == 7 && moved.events()[0].node == recorded,
            "moved snapshots leave no falsely prepared empty event plan");
    Fake fake;
    CudaNativeGraphEffects empty;
    require(prepare_native_graph_effects(fake.graph(), Fake::dispatch(), empty) == CUDA_SUCCESS &&
                empty && empty.node_count() == 0 && empty.events().empty(),
            "empty source graph is a prepared snapshot, not a missing output");
}
void preserved_node_identities() {
    CudaNativeGraphEffects effects;
    std::array<CUgraphNode, 3> supported{};
    CUgraphNode empty{}, recorded{};
    {
        Fake fake;
        const auto graph = fake.graph();
        supported = {fake.node(graph, CU_GRAPH_NODE_TYPE_KERNEL),
                     fake.node(graph, CU_GRAPH_NODE_TYPE_MEMCPY),
                     fake.node(graph, CU_GRAPH_NODE_TYPE_MEMSET)};
        empty = fake.node(graph, CU_GRAPH_NODE_TYPE_EMPTY);
        recorded = fake.node(graph, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
        require(prepare_native_graph_effects(graph, Fake::dispatch(), effects) == CUDA_SUCCESS,
                "retain exact source node types before native construction");
    }
    const std::array types{CU_GRAPH_NODE_TYPE_KERNEL, CU_GRAPH_NODE_TYPE_MEMCPY,
                           CU_GRAPH_NODE_TYPE_MEMSET};
    for (std::size_t index = 0; index < supported.size(); ++index) {
        CudaNativeGraphEffects preserved;
        require(
            effects.preserve_node(supported[index], types[index], preserved) == CUDA_SUCCESS &&
                preserved.node_count() == 5 && preserved.events().size() == 1 &&
                preserved.events()[0].event == kEvent,
            "known non-event parameters preserve exact dependency effects without source queries");
        require(effects.preserve_node(supported[index], std::nullopt, preserved) ==
                    CUDA_ERROR_INVALID_VALUE,
                "occupied preserved output cannot be overwritten");
        preserved = {};
        require(effects.preserve_node(supported[index], std::nullopt, preserved) == CUDA_SUCCESS,
                "enablement is checked against retained supported node type");
    }
    CudaNativeGraphEffects rejected;
    require(
        effects.preserve_node(empty, std::nullopt, rejected) == CUDA_ERROR_NOT_SUPPORTED &&
            effects.preserve_node(recorded, std::nullopt, rejected) == CUDA_ERROR_NOT_SUPPORTED &&
            !rejected,
        "empty and event nodes cannot guess SDK enablement support");
}

void whole_and_child_replacements() {
    CudaNativeGraphEffects original, child_replacement, whole_replacement, ambiguous, ambiguous_new;
    CUgraphNode child_node{}, original_child_record{}, outside{};
    const auto next_record = reinterpret_cast<CUevent>(0x3000);
    const auto next_wait = reinterpret_cast<CUevent>(0x4000);
    {
        Fake fake;
        const auto root = fake.graph();
        const auto child = fake.graph();
        outside = fake.node(root, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kOtherEvent);
        child_node = fake.node(root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, child);
        original_child_record = fake.node(child, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
        fake.node(child, CU_GRAPH_NODE_TYPE_WAIT_EVENT, kOtherEvent);
        fake.node(child, CU_GRAPH_NODE_TYPE_KERNEL);
        const auto next_child = fake.graph();
        fake.node(next_child, CU_GRAPH_NODE_TYPE_KERNEL);
        fake.node(next_child, CU_GRAPH_NODE_TYPE_WAIT_EVENT, next_wait);
        fake.node(next_child, CU_GRAPH_NODE_TYPE_EVENT_RECORD, next_record);
        const auto next_root = fake.graph();
        fake.node(next_root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, next_child);
        fake.node(next_root, CU_GRAPH_NODE_TYPE_EVENT_RECORD, next_wait);
        require(prepare_native_graph_effects(root, Fake::dispatch(), original) == CUDA_SUCCESS &&
                    prepare_native_graph_effects(next_child, Fake::dispatch(), child_replacement) ==
                        CUDA_SUCCESS &&
                    prepare_native_graph_effects(next_root, Fake::dispatch(), whole_replacement) ==
                        CUDA_SUCCESS,
                "snapshot old and new scope identities before any original SDK update");
        const auto siblings = fake.graph();
        fake.node(siblings, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
        fake.node(siblings, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kOtherEvent);
        const auto new_siblings = fake.graph();
        fake.node(new_siblings, CU_GRAPH_NODE_TYPE_EVENT_RECORD, next_record);
        fake.node(new_siblings, CU_GRAPH_NODE_TYPE_EVENT_RECORD, next_wait);
        require(
            prepare_native_graph_effects(siblings, Fake::dispatch(), ambiguous) == CUDA_SUCCESS &&
                prepare_native_graph_effects(new_siblings, Fake::dispatch(), ambiguous_new) ==
                    CUDA_SUCCESS,
            "snapshot ambiguous siblings without inventing creation order");
    }
    CudaNativeGraphEffects child_changed;
    require(original.replace_child(child_node, child_replacement, child_changed) == CUDA_SUCCESS &&
                child_changed.events()[0].node == outside &&
                child_changed.events()[0].event == kOtherEvent &&
                std::ranges::any_of(child_changed.events(),
                                    [&](const auto& effect) {
                                        return effect.node == original_child_record &&
                                               effect.event == next_record;
                                    }) &&
                original.events()[1].event == kEvent,
            "child splice preserves unrelated effects and unique original aliases after source "
            "retirement");
    CudaNativeGraphEffects whole_changed;
    require(original.replace_graph(whole_replacement, whole_changed) == CUDA_SUCCESS &&
                std::ranges::any_of(whole_changed.events(),
                                    [&](const auto& effect) {
                                        return effect.node == outside && effect.event == next_wait;
                                    }),
            "whole replacement maps unique event types by scope, not enumeration order");
    CudaNativeGraphEffects subsequent;
    require(
        whole_changed.replace_child(child_node, child_replacement, subsequent) == CUDA_SUCCESS &&
            std::ranges::any_of(subsequent.events(),
                                [&](const auto& effect) {
                                    return effect.node == outside && effect.event == next_wait;
                                }),
        "whole-to-child replacement keeps original parent identity and unrelated effects");
    CudaNativeGraphEffects uncertain;
    require(
        ambiguous.replace_graph(ambiguous_new, uncertain) == CUDA_SUCCESS &&
            uncertain.events().size() == 2 && !uncertain.events()[0].node &&
            !uncertain.events()[1].node && uncertain.events()[0].event == next_record &&
            uncertain.events()[1].event == next_wait,
        "ambiguous aliases retain complete replay effects without guessed original node identity");
}

void nested_replacements() {
    CudaNativeGraphEffects original, whole, child, leaf, wrong_size;
    CUgraphNode parent{}, nested{}, outside{}, record{};
    const auto next = reinterpret_cast<CUevent>(0x5000);
    const auto final = reinterpret_cast<CUevent>(0x6000);
    {
        Fake fake;
        const auto root = fake.graph();
        const auto inner = fake.graph();
        const auto deepest = fake.graph();
        outside = fake.node(root, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kOtherEvent);
        parent = fake.node(root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, inner);
        fake.node(inner, CU_GRAPH_NODE_TYPE_WAIT_EVENT, kOtherEvent);
        nested = fake.node(inner, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, deepest);
        record = fake.node(deepest, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
        fake.node(deepest, CU_GRAPH_NODE_TYPE_KERNEL);
        const auto new_leaf = fake.graph();
        fake.node(new_leaf, CU_GRAPH_NODE_TYPE_KERNEL);
        fake.node(new_leaf, CU_GRAPH_NODE_TYPE_EVENT_RECORD, next);
        const auto new_child = fake.graph();
        fake.node(new_child, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, new_leaf);
        fake.node(new_child, CU_GRAPH_NODE_TYPE_WAIT_EVENT, next);
        const auto new_root = fake.graph();
        fake.node(new_root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, new_child);
        fake.node(new_root, CU_GRAPH_NODE_TYPE_EVENT_RECORD, final);
        const auto final_leaf = fake.graph();
        fake.node(final_leaf, CU_GRAPH_NODE_TYPE_EVENT_RECORD, final);
        fake.node(final_leaf, CU_GRAPH_NODE_TYPE_KERNEL);
        const auto short_leaf = fake.graph();
        fake.node(short_leaf, CU_GRAPH_NODE_TYPE_EVENT_RECORD, final);
        require(
            prepare_native_graph_effects(root, Fake::dispatch(), original) == CUDA_SUCCESS &&
                prepare_native_graph_effects(new_root, Fake::dispatch(), whole) == CUDA_SUCCESS &&
                prepare_native_graph_effects(new_child, Fake::dispatch(), child) == CUDA_SUCCESS &&
                prepare_native_graph_effects(final_leaf, Fake::dispatch(), leaf) == CUDA_SUCCESS &&
                prepare_native_graph_effects(short_leaf, Fake::dispatch(), wrong_size) ==
                    CUDA_SUCCESS,
            "snapshot reordered nested scopes before destroying every source");
    }
    CudaNativeGraphEffects first, second, third;
    require(original.replace_graph(whole, first) == CUDA_SUCCESS &&
                first.replace_child(parent, child, second) == CUDA_SUCCESS &&
                second.replace_child(nested, leaf, third) == CUDA_SUCCESS,
            "whole-to-parent-to-nested replacement keeps original scope handles");
    require(std::ranges::any_of(third.events(),
                                [&](const auto& effect) {
                                    return effect.node == outside && effect.event == final;
                                }) &&
                std::ranges::any_of(third.events(),
                                    [&](const auto& effect) {
                                        return effect.node == record && effect.event == final;
                                    }) &&
                std::ranges::any_of(third.events(),
                                    [&](const auto& effect) {
                                        return effect.kind == CudaNativeGraphEventKind::kWait &&
                                               effect.event == next;
                                    }) &&
                original.events()[0].event == kOtherEvent,
            "nested splice uses rebound ranges while preserving outer and sibling effects");
    CudaNativeGraphEffects rejected;
    require(third.replace_child(nested, wrong_size, rejected) == CUDA_ERROR_NOT_SUPPORTED &&
                third.replace_child(outside, leaf, rejected) == CUDA_ERROR_INVALID_HANDLE &&
                third.replace_child(nullptr, leaf, rejected) == CUDA_ERROR_INVALID_VALUE &&
                !rejected && third.replace_graph(whole, third) == CUDA_ERROR_INVALID_VALUE,
            "mismatched subtree, non-child, null and occupied outputs publish nothing");
}

void failures() {
    for (const auto fault :
         {Fault::kCount, Fault::kFill, Fault::kVerify, Fault::kType, Fault::kChild, Fault::kRecord,
          Fault::kWait, Fault::kNullNode, Fault::kDuplicateNode, Fault::kUnwrittenCount,
          Fault::kUnwrittenType, Fault::kUnwrittenEvent, Fault::kNullChild, Fault::kGrowing,
          Fault::kShrinking, Fault::kThrow, Fault::kAllocation}) {
        Fake fake;
        const auto root = fake.graph();
        const auto child = fake.graph();
        fake.node(root, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
        fake.node(root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, child);
        fake.node(child, CU_GRAPH_NODE_TYPE_WAIT_EVENT, kOtherEvent);
        fake.fault = fault;
        CudaNativeGraphEffects effects;
        const auto result = prepare_native_graph_effects(root, Fake::dispatch(), effects);
        const bool is_native_failure = fault >= Fault::kCount && fault <= Fault::kWait;
        const auto expected = is_native_failure             ? kMetadataError
                              : fault == Fault::kAllocation ? CUDA_ERROR_OUT_OF_MEMORY
                              : fault == Fault::kGrowing || fault == Fault::kShrinking
                                  ? CUDA_ERROR_NOT_READY
                                  : CUDA_ERROR_UNKNOWN;
        require(result == expected,
                "original getter errors and uncertain output must stay checked");
        require_empty(effects);
    }
}
void bounds_and_dispatch() {
    Fake fake;
    const auto root = fake.graph();
    const auto child = fake.graph();
    fake.node(root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, child);
    fake.node(child, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
    CudaNativeGraphEffects output;
    require(prepare_native_graph_effects(nullptr, Fake::dispatch(), output) ==
                    CUDA_ERROR_INVALID_VALUE &&
                prepare_native_graph_effects(root, Fake::dispatch(), output, {0, 1}) ==
                    CUDA_ERROR_INVALID_VALUE &&
                prepare_native_graph_effects(root, Fake::dispatch(), output, {4097, 1}) ==
                    CUDA_ERROR_INVALID_VALUE &&
                prepare_native_graph_effects(root, Fake::dispatch(), output, {1, 0}) ==
                    CUDA_ERROR_INVALID_VALUE &&
                prepare_native_graph_effects(root, Fake::dispatch(), output, {1, 33}) ==
                    CUDA_ERROR_INVALID_VALUE &&
                fake.calls == 0,
            "invalid bounds reject before any original metadata call");
    require(prepare_native_graph_effects(root, {}, output) == CUDA_ERROR_NOT_SUPPORTED &&
                prepare_native_graph_effects(root, Fake::dispatch(), output, {1, 2}) ==
                    CUDA_ERROR_NOT_SUPPORTED &&
                prepare_native_graph_effects(root, Fake::dispatch(), output, {2, 1}) ==
                    CUDA_ERROR_NOT_SUPPORTED,
            "bounded node/depth exhaustion leaves no partial plan");
    require_empty(output);
    for (unsigned int missing = 0; missing < 5; ++missing) {
        auto dispatch = Fake::dispatch();
        if (missing == 0) {
            dispatch.nodes = nullptr;
        }
        if (missing == 1) {
            dispatch.node_type = nullptr;
        }
        if (missing == 2) {
            dispatch.child_graph = nullptr;
        }
        if (missing == 3) {
            dispatch.recorded_event = nullptr;
        }
        if (missing == 4) {
            dispatch.waited_event = nullptr;
            fake.node(child, CU_GRAPH_NODE_TYPE_WAIT_EVENT, kOtherEvent);
        }
        require(prepare_native_graph_effects(root, dispatch, output) == CUDA_ERROR_NOT_SUPPORTED,
                "missing required original getter fails closed without skipping an effect");
        require_empty(output);
    }
    fake.node(child, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, root);
    require(
        prepare_native_graph_effects(root, Fake::dispatch(), output) == CUDA_ERROR_INVALID_VALUE,
        "native metadata cycles cannot cause unbounded recursion");
    require_empty(output);
}
void unsupported_and_repeated_nodes() {
    for (const auto type : {CU_GRAPH_NODE_TYPE_HOST, CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL,
                            CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT, CU_GRAPH_NODE_TYPE_MEM_ALLOC,
                            CU_GRAPH_NODE_TYPE_MEM_FREE, static_cast<CUgraphNodeType>(12),
                            static_cast<CUgraphNodeType>(13), static_cast<CUgraphNodeType>(14)}) {
        Fake fake;
        const auto graph = fake.graph();
        fake.node(graph, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
        fake.node(graph, type);
        CudaNativeGraphEffects effects;
        require(prepare_native_graph_effects(graph, Fake::dispatch(), effects) ==
                    CUDA_ERROR_NOT_SUPPORTED,
                "unsupported effects cannot masquerade as a complete graph plan");
        require_empty(effects);
    }
    Fake fake;
    const auto root = fake.graph();
    const auto child = fake.graph();
    const auto record = fake.node(child, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
    fake.node(root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, child);
    fake.node(root, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, child);
    CudaNativeGraphEffects effects;
    require(prepare_native_graph_effects(root, Fake::dispatch(), effects, {4, 2}) == CUDA_SUCCESS &&
                effects.node_count() == 4 && effects.events().size() == 2 &&
                effects.events()[0].node == record && effects.events()[1].node == record,
            "repeated child references are counted per traversal, not mistaken for path cycles");
}
void exact_bounds_and_same_event_roles() {
    Fake fake;
    const auto root = fake.graph();
    const auto record = fake.node(root, CU_GRAPH_NODE_TYPE_EVENT_RECORD, kEvent);
    const auto wait = fake.node(root, CU_GRAPH_NODE_TYPE_WAIT_EVENT, kEvent);
    CudaNativeGraphEffects roles;
    require(prepare_native_graph_effects(root, Fake::dispatch(), roles, {2, 1}) == CUDA_SUCCESS &&
                roles.events().size() == 2 && roles.events()[0].node == record &&
                roles.events()[1].node == wait &&
                roles.events()[0].event == roles.events()[1].event &&
                roles.events()[0].kind != roles.events()[1].kind,
            "identical event handles do not merge distinct record/wait effects");
    const auto deep = fake.graph();
    auto parent = deep;
    for (unsigned int level = 1; level < 32; ++level) {
        const auto child = fake.graph();
        fake.node(parent, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, child);
        parent = child;
    }
    fake.node(parent, CU_GRAPH_NODE_TYPE_EMPTY);
    CudaNativeGraphEffects boundary;
    require(
        prepare_native_graph_effects(deep, Fake::dispatch(), boundary, {32, 32}) == CUDA_SUCCESS &&
            boundary.node_count() == 32 && boundary.events().empty(),
        "exact supported depth/node bounds remain usable");
    const auto beyond = fake.graph();
    fake.node(parent, CU_GRAPH_NODE_TYPE_GRAPH, nullptr, beyond);
    CudaNativeGraphEffects rejected;
    require(
        prepare_native_graph_effects(deep, Fake::dispatch(), rejected) == CUDA_ERROR_NOT_SUPPORTED,
        "one extra nested graph cannot cross the traversal depth bound");
    require_empty(rejected);
    const auto plain = fake.graph();
    fake.node(plain, CU_GRAPH_NODE_TYPE_KERNEL);
    CudaNativeGraphEffects kernel_only;
    require(prepare_native_graph_effects(plain, {Fake::get_nodes, Fake::get_type, {}, {}, {}},
                                         kernel_only) == CUDA_SUCCESS &&
                kernel_only.node_count() == 1 && kernel_only.events().empty(),
            "optional getters are not required for effects absent from the graph");
}
}  // namespace
int main() {
    try {
        require(native_graph_flags_allow_effect_preparation(0),
                "zero flags permit complete discovery");
#if CUDA_VERSION >= 11040
        require(native_graph_flags_allow_effect_preparation(
                    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH),
                "AutoFree may prepare only the existing memory-node-free effects whitelist");
#endif
#if CUDA_VERSION >= 12000
        require(native_graph_flags_allow_effect_preparation(
                    CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY) &&
                    native_graph_flags_allow_effect_preparation(
                        CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY |
                        CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH),
                "original node priorities and memory-node-free AutoFree can coexist");
#endif
        for (const cuuint64_t flags : {2ULL, 4ULL, 7ULL, 16ULL, 1ULL << 63})
            require(!native_graph_flags_allow_effect_preparation(flags),
                    "upload, device launch and unknown bits remain unprepared");
        identities_and_lifetime();
        preserved_node_identities();
        whole_and_child_replacements();
        nested_replacements();
        failures();
        bounds_and_dispatch();
        unsupported_and_repeated_nodes();
        exact_bounds_and_same_event_roles();
        std::array<std::thread, 3> callers;
        for (auto& caller : callers) {
            caller = std::thread(identities_and_lifetime);
        }
        for (auto& caller : callers) {
            caller.join();
        }
        std::cout << "native graph event effects and transactional discovery passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
