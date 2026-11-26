#include "internal/cuda_execution_graph.h"
#include "internal/cuda_native_output_cell.h"

#include <algorithm>
#include <bit>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> graph_owner_thread;
struct GraphHealth {
    pid_t process{::getpid()};
    std::weak_ptr<const void> thread{graph_owner_thread};
    CUcontext context{};
    decltype(&cuCtxGetCurrent) current{};
    bool is_active{}, is_unknown{}, has_capture{};
    CUresult gate() const noexcept {
        if (::getpid() != process || !graph_owner_thread || thread.lock() != graph_owner_thread)
            return CUDA_ERROR_INVALID_CONTEXT;
        if (is_unknown) return CUDA_ERROR_UNKNOWN;
        return is_active ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
    }
    CUresult prove() noexcept {
        CUcontext observed{};
        const auto status = current(&observed);
        if (status != CUDA_SUCCESS || observed != context) {
            is_unknown = true;
            return status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : status;
        }
        return CUDA_SUCCESS;
    }
};
struct GraphScope {
    GraphHealth& health;
    explicit GraphScope(GraphHealth& owner) : health(owner) {
        health.is_active = true;
    }
    ~GraphScope() {
        health.is_active = false;
    }
    GraphScope(const GraphScope&) = delete;
    GraphScope& operator=(const GraphScope&) = delete;
};
struct GraphResources {
    std::vector<std::shared_ptr<const void>> leases;
};
struct GraphObject {
    std::uint64_t id{};
    cudaGraph_t native{};
    std::shared_ptr<const GraphResources> resources;
    bool has_destroy_attempt{};
};
struct GraphExecutable {
    std::uint64_t id{};
    cudaGraphExec_t native{};
    std::shared_ptr<const GraphResources> resources;
    bool has_destroy_attempt{};
};
class GraphLaunch final : public CudaSessionGraphLaunch {
   public:
    GraphLaunch(std::shared_ptr<GraphHealth> health, std::shared_ptr<GraphExecutable> executable,
                decltype(&cuGraphLaunch) launch)
        : health_(std::move(health)), executable_(std::move(executable)), launch_(launch) {}
    std::int32_t issue(CUstream stream, std::span<const std::byte>) noexcept override {
        const auto entry = health_->gate();
        if (entry != CUDA_SUCCESS || health_->has_capture || has_issue_ ||
            executable_->has_destroy_attempt) {
            completion_ = {.status = entry != CUDA_SUCCESS ? entry : CUDA_ERROR_INVALID_VALUE,
                           .was_native_issued = false,
                           .is_clean_preparation_rejection = true};
            return 0;
        }
        const GraphScope scope(*health_);
        auto proof = health_->prove();
        if (proof != CUDA_SUCCESS) {
            completion_ = {
                .status = proof, .is_cleanup_uncertain = true, .was_native_issued = false};
            return 0;
        }
        has_issue_ = true;
        static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
        const auto original = launch_(executable_->native, stream);
        proof = health_->prove();
        if (proof != CUDA_SUCCESS || original != CUDA_SUCCESS) health_->is_unknown = true;
        result_ = CudaSessionGraphResult{.status = proof,
                                         .original_status = static_cast<std::int32_t>(original),
                                         .is_quarantined = health_->is_unknown};
        completion_ = {.status = proof, .is_cleanup_uncertain = health_->is_unknown};
        return original;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }
    std::optional<CudaSessionGraphResult> reply() const noexcept override {
        return result_;
    }
    CUresult cleanup_after_physical_stop() noexcept override {
        executable_.reset();
        return CUDA_SUCCESS;
    }

   private:
    std::shared_ptr<GraphHealth> health_;
    std::shared_ptr<GraphExecutable> executable_;
    decltype(&cuGraphLaunch) launch_;
    bool has_issue_{};
    CudaExecutionLibraryCompletion completion_{.was_native_issued = false};
    std::optional<CudaSessionGraphResult> result_;
};
}  // namespace

struct CudaExecutionGraph::Implementation {
    const CudaSessionGraphDispatch dispatch;
    const std::size_t maximum_objects, maximum_nodes;
    std::shared_ptr<GraphHealth> health;
    std::vector<std::shared_ptr<GraphObject>> graphs;
    std::vector<std::shared_ptr<GraphExecutable>> executables;
    std::uint64_t next_id{1}, stream{}, caller{};
    CUstream native_stream{};
    int mode{};
    bool is_invalidated{};
    cudaStreamCaptureStatus capture_status{cudaStreamCaptureStatusNone};
    std::shared_ptr<GraphResources> resources;
    // Independent object and per-capture node bounds are checked before SDK entry.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    Implementation(CudaSessionGraphDispatch functions, CUcontext context,
                   decltype(&cuCtxGetCurrent) current, std::size_t objects, std::size_t nodes)
        : dispatch(functions),
          maximum_objects(objects),
          maximum_nodes(nodes),
          health(std::make_shared<GraphHealth>()) {
        health->context = context;
        health->current = current;
        graphs.reserve(objects);
        executables.reserve(objects);
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)
    CudaSessionGraphResult result(CUresult status = CUDA_SUCCESS) const noexcept {
        return {.status = status,
                .capture_status = static_cast<std::uint32_t>(
                    is_invalidated && health->has_capture ? cudaStreamCaptureStatusInvalidated
                                                          : capture_status),
                .is_quarantined = health->is_unknown};
    }
    CudaSessionGraphResult unknown(CUresult status = CUDA_ERROR_UNKNOWN) noexcept {
        health->is_unknown = true;
        return result(status);
    }
    CUresult matching(std::uint64_t requested_stream,
                      std::uint64_t requested_caller) const noexcept {
        if (!health->has_capture || requested_stream != stream || !requested_caller)
            return CUDA_ERROR_INVALID_HANDLE;
        if (mode != cudaStreamCaptureModeRelaxed && requested_caller != caller)
            return CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
        return CUDA_SUCCESS;
    }
    CUresult observe() noexcept {
        cudaStreamCaptureStatus observed{};
        const auto original = dispatch.state(native_stream, &observed);
        const auto proof = health->prove();
        if (original != cudaSuccess || proof != CUDA_SUCCESS ||
            (observed != cudaStreamCaptureStatusNone && observed != cudaStreamCaptureStatusActive &&
             observed != cudaStreamCaptureStatusInvalidated)) {
            health->is_unknown = true;
            return proof == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : proof;
        }
        capture_status = observed;
        return CUDA_SUCCESS;
    }
    bool has_capacity() const noexcept {
        return next_id && graphs.size() + executables.size() < maximum_objects;
    }
    std::uint64_t take_id() noexcept {
        const auto id = next_id;
        next_id = next_id == UINT64_MAX ? 0 : next_id + 1;
        return id;
    }
};

CudaExecutionGraph::CudaExecutionGraph(CudaSessionGraphDispatch dispatch, CUcontext context,
                                       decltype(&cuCtxGetCurrent) current,
                                       std::size_t maximum_objects, std::size_t maximum_nodes) {
    if (!dispatch.begin || !dispatch.end || !dispatch.state || !dispatch.instantiate ||
        !dispatch.destroy || !dispatch.destroy_exec || !dispatch.launch || !context || !current ||
        !maximum_objects || !maximum_nodes || maximum_nodes == SIZE_MAX)
        throw std::invalid_argument("Incomplete graph owner configuration");
    if (!graph_owner_thread) graph_owner_thread = std::make_shared<const int>(0);
    implementation_ = std::make_unique<Implementation>(dispatch, context, current, maximum_objects,
                                                       maximum_nodes);
}
CudaExecutionGraph::~CudaExecutionGraph() {
    if (implementation_ &&
        (implementation_->health->gate() != CUDA_SUCCESS || implementation_->health->has_capture ||
         !implementation_->graphs.empty() || !implementation_->executables.empty()))
        (void)implementation_.release();
}
CudaSessionGraphSnapshot CudaExecutionGraph::snapshot() const noexcept {
    const auto& state = *implementation_;
    const auto status = state.health->gate();
    if (status == CUDA_ERROR_INVALID_CONTEXT || status == CUDA_ERROR_NOT_READY)
        return {.status = status};
    return {.status = status,
            .stream = state.stream,
            .caller = state.caller,
            .graphs = state.graphs.size(),
            .executables = state.executables.size(),
            .has_capture = state.health->has_capture,
            .is_unknown = state.health->is_unknown,
            .is_invalidated = state.is_invalidated};
}
CudaSessionGraphResult CudaExecutionGraph::begin(std::uint64_t stream, CUstream native, int mode,
                                                 std::uint64_t caller,
                                                 std::shared_ptr<const void> lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (state.health->has_capture || !stream || !native || !caller || !lifetime ||
        mode < cudaStreamCaptureModeGlobal || mode > cudaStreamCaptureModeRelaxed)
        return state.result(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_capacity()) return state.result(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        auto resources = std::make_shared<GraphResources>();
        resources->leases.reserve(state.maximum_nodes + 1);
        resources->leases.push_back(std::move(lifetime));
        const GraphScope scope(*state.health);
        if (const auto proof = state.health->prove(); proof != CUDA_SUCCESS)
            return state.result(proof);
        state.stream = stream;
        state.native_stream = native;
        state.caller = caller;
        state.mode = mode;
        state.is_invalidated = false;
        state.resources = std::move(resources);
        // A failed or interrupted Begin can still have effects; retain ownership
        // before entering the SDK and release it only after an independent query.
        state.health->has_capture = true;
        const auto original =
            state.dispatch.begin(native, static_cast<cudaStreamCaptureMode>(mode));
        auto proof = state.health->prove();
        if (proof == CUDA_SUCCESS) proof = state.observe();
        if (proof == CUDA_SUCCESS && state.capture_status == cudaStreamCaptureStatusNone) {
            if (original == cudaSuccess)
                proof = state.unknown().status;
            else {
                state.health->has_capture = false;
                state.resources.reset();
            }
        }
        auto result = state.result(proof);
        result.original_status = static_cast<std::int32_t>(original);
        return result;
    } catch (...) {
        if (state.health->has_capture) return state.unknown();
        return state.result(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionGraphResult CudaExecutionGraph::state(std::uint64_t stream,
                                                 std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (const auto status = state.matching(stream, caller); status != CUDA_SUCCESS)
        return state.result(status);
    const GraphScope scope(*state.health);
    if (const auto proof = state.health->prove(); proof != CUDA_SUCCESS) return state.result(proof);
    cudaStreamCaptureStatus observed{};
    const auto original = state.dispatch.state(state.native_stream, &observed);
    auto proof = state.health->prove();
    if (proof == CUDA_SUCCESS && original == cudaSuccess) {
        if (observed != cudaStreamCaptureStatusActive &&
            observed != cudaStreamCaptureStatusInvalidated)
            proof = state.unknown().status;
        else
            state.capture_status = observed;
    }
    auto result = state.result(proof);
    result.original_status = static_cast<std::int32_t>(original);
    return result;
}
CudaSessionGraphResult CudaExecutionGraph::append(std::uint64_t stream, std::uint64_t caller,
                                                  std::shared_ptr<const void> lifetime,
                                                  const std::function<CUresult()>& issue) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (const auto status = state.matching(stream, caller); status != CUDA_SUCCESS)
        return state.result(status);
    if (state.is_invalidated || state.capture_status == cudaStreamCaptureStatusInvalidated)
        return state.result(CUDA_ERROR_STREAM_CAPTURE_INVALIDATED);
    if (!lifetime || !issue) return state.result(CUDA_ERROR_INVALID_VALUE);
    if (state.resources->leases.size() > state.maximum_nodes)
        return state.result(CUDA_ERROR_OUT_OF_MEMORY);
    const GraphScope scope(*state.health);
    if (const auto proof = state.health->prove(); proof != CUDA_SUCCESS) return state.result(proof);
    state.resources->leases.push_back(std::move(lifetime));
    CUresult original;
    try {
        original = issue();
    } catch (...) {
        return state.unknown();
    }
    auto proof = state.health->prove();
    if (proof == CUDA_SUCCESS) proof = state.observe();
    if (proof == CUDA_SUCCESS && state.capture_status == cudaStreamCaptureStatusNone)
        proof = state.unknown().status;
    auto result = state.result(proof);
    result.original_status = static_cast<std::int32_t>(original);
    return result;
}
CudaSessionGraphResult CudaExecutionGraph::invalidate(std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (const auto status = state.matching(state.stream, caller); status != CUDA_SUCCESS)
        return state.result(status);
    state.is_invalidated = true;
    return state.result(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
}
CudaSessionGraphResult CudaExecutionGraph::end(std::uint64_t stream, std::uint64_t caller,
                                               bool has_output, std::uint64_t seed) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (const auto status = state.matching(stream, caller); status != CUDA_SUCCESS)
        return state.result(status);
    try {
        auto output = CudaNativeOutputCell::create(std::as_bytes(std::span{&seed, 1}));
        if (output.status != CUDA_SUCCESS) return state.result(output.status);
        auto graph = std::make_shared<GraphObject>();
        graph->resources = state.resources;
        const GraphScope scope(*state.health);
        if (const auto proof = state.health->prove(); proof != CUDA_SUCCESS)
            return state.result(proof);
        const auto original = state.dispatch.end(
            state.native_stream,
            has_output ? static_cast<cudaGraph_t*>(output.cell->data()) : nullptr);
        auto result = state.result(state.health->prove());
        result.original_status = static_cast<std::int32_t>(original);
        if (has_output) {
            const auto captured =
                output.cell->snapshot(std::as_writable_bytes(std::span{&result.output, 1}));
            result.output_written = captured.status == CUDA_SUCCESS && captured.was_written;
            if (captured.status != CUDA_SUCCESS)
                result.status = state.unknown(captured.status).status;
        }
        if (result.status == CUDA_SUCCESS) result.status = state.observe();
        if (result.output_written) {
            graph->native = std::bit_cast<cudaGraph_t>(result.output);
        }
        const bool ended =
            result.status == CUDA_SUCCESS && state.capture_status == cudaStreamCaptureStatusNone;
        if (ended) state.health->has_capture = false;
        if (original == cudaSuccess &&
            (!ended || !has_output || !result.output_written || !graph->native))
            result.status = state.unknown().status;
        if (original != cudaSuccess && graph->native) result.status = state.unknown().status;
        if (result.status == CUDA_SUCCESS && ended && state.is_invalidated && graph->native) {
            graph->has_destroy_attempt = true;
            const auto destroyed = state.dispatch.destroy(graph->native);
            const auto proof = state.health->prove();
            if (destroyed != cudaSuccess || proof != CUDA_SUCCESS)
                result.status =
                    state.unknown(proof == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : proof).status;
            else {
                graph->native = nullptr;
                result.output = 0;
            }
        }
        if (graph->native) {
            graph->id = state.take_id();
            state.graphs.push_back(graph);
            if (result.status == CUDA_SUCCESS && original == cudaSuccess && !state.is_invalidated)
                result.id = graph->id;
        }
        if (ended && !state.health->is_unknown) state.resources.reset();
        if (result.status == CUDA_SUCCESS && ended && state.is_invalidated) {
            result.status = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
            result.output_written = has_output;
            result.output = 0;
        }
        result.capture_status = static_cast<std::uint32_t>(state.capture_status);
        result.is_quarantined = state.health->is_unknown;
        return result;
    } catch (...) {
        return state.unknown(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionGraphResult CudaExecutionGraph::abort() noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (!state.health->has_capture) return state.result();
    state.is_invalidated = true;
    return end(state.stream, state.caller, true, 0);
}
// Logical object identity and original SDK flags retain separate checked fields.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CudaSessionGraphResult CudaExecutionGraph::instantiate(std::uint64_t id, std::uint64_t flags,
                                                       bool has_output,
                                                       std::uint64_t seed) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (state.health->has_capture) return state.result(CUDA_ERROR_NOT_READY);
    const auto found =
        std::ranges::find_if(state.graphs, [id](const auto& graph) { return graph->id == id; });
    if (found == state.graphs.end() || (*found)->has_destroy_attempt)
        return state.result(CUDA_ERROR_INVALID_HANDLE);
    if (!state.has_capacity()) return state.result(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        auto output = CudaNativeOutputCell::create(std::as_bytes(std::span{&seed, 1}));
        if (output.status != CUDA_SUCCESS) return state.result(output.status);
        auto executable = std::make_shared<GraphExecutable>();
        executable->resources = (*found)->resources;
        const GraphScope scope(*state.health);
        if (const auto proof = state.health->prove(); proof != CUDA_SUCCESS)
            return state.result(proof);
        const auto original = state.dispatch.instantiate(
            has_output ? static_cast<cudaGraphExec_t*>(output.cell->data()) : nullptr,
            (*found)->native, flags);
        auto result = state.result(state.health->prove());
        result.original_status = static_cast<std::int32_t>(original);
        if (has_output) {
            const auto captured =
                output.cell->snapshot(std::as_writable_bytes(std::span{&result.output, 1}));
            result.output_written = captured.status == CUDA_SUCCESS && captured.was_written;
            if (captured.status != CUDA_SUCCESS)
                result.status = state.unknown(captured.status).status;
        }
        if (result.output_written)
            executable->native = std::bit_cast<cudaGraphExec_t>(result.output);
        if ((original == cudaSuccess &&
             (!has_output || !result.output_written || !executable->native)) ||
            (original != cudaSuccess && executable->native))
            result.status = state.unknown().status;
        if (executable->native) {
            executable->id = state.take_id();
            state.executables.push_back(executable);
            if (result.status == CUDA_SUCCESS && original == cudaSuccess)
                result.id = executable->id;
        }
        result.is_quarantined = state.health->is_unknown;
        return result;
    } catch (...) {
        return state.unknown(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionGraphResult CudaExecutionGraph::destroy(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (state.health->has_capture) return state.result(CUDA_ERROR_NOT_READY);
    const auto found =
        std::ranges::find_if(state.graphs, [id](const auto& graph) { return graph->id == id; });
    if (found == state.graphs.end() || (*found)->has_destroy_attempt)
        return state.result(CUDA_ERROR_INVALID_HANDLE);
    const GraphScope scope(*state.health);
    if (const auto proof = state.health->prove(); proof != CUDA_SUCCESS) return state.result(proof);
    (*found)->has_destroy_attempt = true;
    const auto original = state.dispatch.destroy((*found)->native);
    auto result = state.result(state.health->prove());
    result.original_status = static_cast<std::int32_t>(original);
    if (result.status == CUDA_SUCCESS && original == cudaSuccess)
        state.graphs.erase(found);
    else
        result.status =
            state.unknown(result.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result.status)
                .status;
    result.is_quarantined = state.health->is_unknown;
    return result;
}
CudaSessionGraphResult CudaExecutionGraph::destroy_exec(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS)
        return state.result(status);
    if (state.health->has_capture) return state.result(CUDA_ERROR_NOT_READY);
    const auto found = std::ranges::find_if(state.executables,
                                            [id](const auto& value) { return value->id == id; });
    if (found == state.executables.end() || (*found)->has_destroy_attempt)
        return state.result(CUDA_ERROR_INVALID_HANDLE);
    if (found->use_count() != 1) return state.result(CUDA_ERROR_NOT_READY);
    const GraphScope scope(*state.health);
    if (const auto proof = state.health->prove(); proof != CUDA_SUCCESS) return state.result(proof);
    (*found)->has_destroy_attempt = true;
    const auto original = state.dispatch.destroy_exec((*found)->native);
    auto result = state.result(state.health->prove());
    result.original_status = static_cast<std::int32_t>(original);
    if (result.status == CUDA_SUCCESS && original == cudaSuccess)
        state.executables.erase(found);
    else
        result.status =
            state.unknown(result.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result.status)
                .status;
    result.is_quarantined = state.health->is_unknown;
    return result;
}
CudaSessionGraphLaunchPreparation CudaExecutionGraph::prepare_launch(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS) return {status, {}};
    if (state.health->has_capture) return {CUDA_ERROR_NOT_READY, {}};
    const auto found = std::ranges::find_if(state.executables,
                                            [id](const auto& value) { return value->id == id; });
    if (found == state.executables.end() || (*found)->has_destroy_attempt) return {};
    try {
        return {CUDA_SUCCESS,
                std::make_shared<GraphLaunch>(state.health, *found, state.dispatch.launch)};
    } catch (...) {
        return {CUDA_ERROR_OUT_OF_MEMORY, {}};
    }
}
CUresult CudaExecutionGraph::shutdown() noexcept {
    auto& state = *implementation_;
    if (const auto status = state.health->gate(); status != CUDA_SUCCESS) return status;
    if (state.health->has_capture) return CUDA_ERROR_NOT_READY;
    while (!state.graphs.empty()) {
        const auto result = destroy(state.graphs.back()->id);
        if (result.status != CUDA_SUCCESS) return result.status;
    }
    while (!state.executables.empty()) {
        const auto result = destroy_exec(state.executables.back()->id);
        if (result.status != CUDA_SUCCESS) return result.status;
    }
    return CUDA_SUCCESS;
}

}  // namespace glimmer::backend::cuda
