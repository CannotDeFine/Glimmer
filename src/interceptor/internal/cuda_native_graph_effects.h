#pragma once

#include <cuda.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace glimmer::interceptor {

// Flags alone never authorize replay. The complete effects whitelist must still
// exclude memory allocation/free nodes and unsupported effects recursively.
[[nodiscard]] constexpr bool native_graph_flags_allow_effect_preparation(
    cuuint64_t flags) noexcept {
#if CUDA_VERSION >= 12000
    constexpr cuuint64_t k_supported_flags = CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH |
                                             CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
#elif CUDA_VERSION >= 11040
    constexpr cuuint64_t k_supported_flags = CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
#else
    constexpr cuuint64_t k_supported_flags = 0;
#endif
    return (flags & ~k_supported_flags) == 0;
}

struct CudaNativeGraphMetadataDispatch {
    decltype(&cuGraphGetNodes) nodes{};
    decltype(&cuGraphNodeGetType) node_type{};
    decltype(&cuGraphChildGraphNodeGetGraph) child_graph{};
    decltype(&cuGraphEventRecordNodeGetEvent) recorded_event{};
    decltype(&cuGraphEventWaitNodeGetEvent) waited_event{};
};
struct CudaNativeGraphEffectsLimits {
    std::size_t nodes{4096};
    std::size_t depth{32};
};
enum class CudaNativeGraphEventKind { kRecord, kWait };
struct CudaNativeGraphEventEffect {
    // NULL after a replacement with ambiguous original-node correspondence.
    // The event/kind still describe a complete replay effect, not a setter alias.
    CUgraphNode node{};
    CUevent event{};
    CudaNativeGraphEventKind kind{};
};

// Immutable CPU snapshot. Handles identify original nodes/events, not owning
// CUDA references. Never query a destroyed source node through this snapshot.
class CudaNativeGraphEffects final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return data_ != nullptr;
    }
    [[nodiscard]] std::size_t node_count() const noexcept {
        return data_ ? data_->node_count : 0;
    }
    [[nodiscard]] std::span<const CudaNativeGraphEventEffect> events() const noexcept {
        return data_ ? std::span<const CudaNativeGraphEventEffect>{data_->events}
                     : std::span<const CudaNativeGraphEventEffect>{};
    }
    // CPU-only immutable replacement; no source graph/node lifetime is borrowed.
    // Unknown/ambiguous node or kind rejects and leaves output empty.
    [[nodiscard]] CUresult replace_event(CUgraphNode node, CUevent event,
                                         CudaNativeGraphEventKind kind,
                                         CudaNativeGraphEffects& output) const noexcept;
    // Only known kernel/memcpy/memset changes preserve event dependencies.
    // An absent expected type denotes SDK node enablement, not a guessed type.
    [[nodiscard]] CUresult preserve_node(CUgraphNode node,
                                         std::optional<CUgraphNodeType> expected_type,
                                         CudaNativeGraphEffects& output) const noexcept;

    // CPU-only staging. Publish only after native update success proves topology.
    // Keep aliases only for unique node types in uniquely matched graph scopes;
    // never infer correspondence from metadata enumeration order.
    [[nodiscard]] CUresult replace_graph(const CudaNativeGraphEffects& replacement,
                                         CudaNativeGraphEffects& output) const noexcept;
    [[nodiscard]] CUresult replace_child(CUgraphNode node,
                                         const CudaNativeGraphEffects& replacement,
                                         CudaNativeGraphEffects& output) const noexcept;

   private:
    friend CUresult prepare_native_graph_effects(CUgraph, const CudaNativeGraphMetadataDispatch&,
                                                 CudaNativeGraphEffects&,
                                                 CudaNativeGraphEffectsLimits) noexcept;
    struct Child {
        CUgraphNode node{};
        std::size_t node_begin{}, node_end{}, event_begin{}, event_end{};
        std::vector<CUgraphNode> ancestors;
    };
    struct Data {
        std::size_t node_count{};
        std::vector<CudaNativeGraphEventEffect> events;
        std::vector<std::pair<CUgraphNode, CUgraphNodeType>> nodes;
        std::vector<CUgraphNode> scopes;
        std::vector<Child> children;
    };
    [[nodiscard]] std::vector<CUgraphNode> replacement_aliases(const Data& replacement,
                                                               CUgraphNode original_scope) const;
    void rebind_children(Data& output, const Data& replacement,
                         const std::vector<CUgraphNode>& aliases, std::size_t event_offset) const;
    std::shared_ptr<const Data> data_;
};

// Original metadata dispatch only. Caller owns actual context/graph lifetime
// and excludes graph mutation/destruction. No native launch or private GPU work.
// Failure leaves output empty; missing dispatch/unsupported effects reject.
[[nodiscard]] CUresult prepare_native_graph_effects(
    CUgraph graph, const CudaNativeGraphMetadataDispatch& dispatch, CudaNativeGraphEffects& output,
    CudaNativeGraphEffectsLimits limits = {}) noexcept;

}  // namespace glimmer::interceptor
