#include "dnn_convolution_fixture.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string_view>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
namespace driver_fixture = glimmer::tests::dnn_convolution;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
enum class Phase {
    kNone,
    kContext,
    kVersion,
    kLayout,
    kOperation,
    kPreference,
    kHeuristic,
    kConfiguration,
    kAlgorithmCheck
};
enum class Route { kAdmission, kHeuristic, kAlgorithmCheck };
enum class Fault { kNone, kContextLoss, kContextLossAndNativeError };
struct Case {
    std::string_view name;
    Route route;
    Phase phase;
};
constexpr std::array kCases{
    Case{"admission-context", Route::kAdmission, Phase::kContext},
    Case{"admission-version", Route::kAdmission, Phase::kVersion},
    Case{"admission-layout", Route::kAdmission, Phase::kLayout},
    Case{"admission-operation", Route::kAdmission, Phase::kOperation},
    Case{"admission-configuration", Route::kAdmission, Phase::kConfiguration},
    Case{"admission-algorithm-check", Route::kAdmission, Phase::kAlgorithmCheck},
    Case{"heuristic-context", Route::kHeuristic, Phase::kContext},
    Case{"heuristic-version", Route::kHeuristic, Phase::kVersion},
    Case{"heuristic-preference", Route::kHeuristic, Phase::kPreference},
    Case{"heuristic-original", Route::kHeuristic, Phase::kHeuristic},
    Case{"algorithm-context", Route::kAlgorithmCheck, Phase::kContext},
    Case{"algorithm-version", Route::kAlgorithmCheck, Phase::kVersion},
    Case{"algorithm-original", Route::kAlgorithmCheck, Phase::kAlgorithmCheck},
};
struct Handle {
    bool alive{};
    CUcontext context{};
};
struct Object {
    CudaLtObjectKind kind{};
};
struct State {
    Handle handle;
    unsigned int calls{}, native_calls{}, context_sets{}, destroys{}, matmuls{}, probes{};
    Phase armed{Phase::kNone};
    std::function<void()> probe;
    std::string_view case_name;
    cublasStatus_t query_status{CUBLAS_STATUS_SUCCESS};
} state;

void touch(Phase phase) {
    ++state.calls;
    if (phase != Phase::kContext) ++state.native_calls;
    if (phase == state.armed && phase != Phase::kNone) {
        state.armed = Phase::kNone;
        ++state.probes;
        state.probe();
    }
}
// Stop immediately on an unsafe callback result. Continuing the RED control
// after successful reentrant destruction could dereference a freed cached owner.
void callback_expect(bool condition, const char* message) {
    if (condition) return;
    std::cerr << state.case_name << ": " << message << '\n';
    std::_Exit(EXIT_FAILURE);
}
void valid_handle(cublasLtHandle_t handle) {
    expect(handle == reinterpret_cast<cublasLtHandle_t>(&state.handle) && state.handle.alive &&
               state.handle.context == driver_fixture::current(),
           "original callback uses the same live owned handle/context");
}
void valid_object(void* pointer, CudaLtObjectKind kind) {
    expect(pointer && static_cast<Object*>(pointer)->kind == kind,
           "original callback receives a typed immutable metadata owner");
}
template <typename T>
cublasStatus_t publish(void* output, std::size_t size, std::size_t* written, T value) {
    expect(output && written && size == sizeof(value), "exact original attribute ABI width");
    std::memcpy(output, &value, sizeof(value));
    *written = sizeof(value);
    return CUBLAS_STATUS_SUCCESS;
}
CudaLtObjectView object(CudaLtObjectKind kind) {
    auto storage = std::make_shared<Object>(Object{kind});
    return {.kind = kind, .native = storage.get(), .bias = {}, .lifetime = std::move(storage)};
}

CudaSessionDispatch dispatch() {
    auto sdk = driver_fixture::session_dispatch();
    sdk.execution.context_get_current = [](CUcontext* output) {
        touch(Phase::kContext);
        *output = driver_fixture::current();
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext context) {
        ++state.context_sets;
        expect(!context || driver_fixture::driver.contexts.contains(context),
               "context switch stays inside the fake Driver");
        driver_fixture::driver.current = context;
        return CUDA_SUCCESS;
    };
    sdk.lt.create = [](cublasLtHandle_t* output) {
        expect(!state.handle.alive && driver_fixture::current(), "one owned Lt handle");
        state.handle = {true, driver_fixture::current()};
        *output = reinterpret_cast<cublasLtHandle_t>(&state.handle);
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.lt.destroy = [](cublasLtHandle_t handle) {
        valid_handle(handle);
        ++state.destroys;
        state.handle.alive = false;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.lt.version = []() -> std::size_t {
        touch(Phase::kVersion);
        return 120604;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected SDK signatures.
    sdk.lt.layout_get = [](cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attribute,
                           void* output, std::size_t bytes, std::size_t* written) {
        valid_object(layout, CudaLtObjectKind::kLayout);
        touch(Phase::kLayout);
        switch (attribute) {
            case CUBLASLT_MATRIX_LAYOUT_TYPE:
                return publish(output, bytes, written, CUDA_R_32F);
            case CUBLASLT_MATRIX_LAYOUT_ORDER:
                return publish(output, bytes, written, CUBLASLT_ORDER_COL);
            case CUBLASLT_MATRIX_LAYOUT_ROWS:
            case CUBLASLT_MATRIX_LAYOUT_COLS:
                return publish(output, bytes, written, std::uint64_t{2});
            case CUBLASLT_MATRIX_LAYOUT_LD:
                return publish(output, bytes, written, std::int64_t{2});
            case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
                return publish(output, bytes, written, std::int32_t{1});
            case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
            case CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET:
                return publish(output, bytes, written, std::int64_t{0});
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    };
    sdk.lt.operation_get = [](cublasLtMatmulDesc_t operation,
                              cublasLtMatmulDescAttributes_t attribute, void* output,
                              std::size_t bytes, std::size_t* written) {
        valid_object(operation, CudaLtObjectKind::kOperation);
        touch(Phase::kOperation);
        switch (attribute) {
            case CUBLASLT_MATMUL_DESC_COMPUTE_TYPE:
                return publish(output, bytes, written, CUBLAS_COMPUTE_32F);
            case CUBLASLT_MATMUL_DESC_SCALE_TYPE:
                return publish(output, bytes, written, CUDA_R_32F);
            case CUBLASLT_MATMUL_DESC_POINTER_MODE:
                return publish(output, bytes, written, CUBLASLT_POINTER_MODE_HOST);
            case CUBLASLT_MATMUL_DESC_EPILOGUE:
                return publish(output, bytes, written, CUBLASLT_EPILOGUE_DEFAULT);
            case CUBLASLT_MATMUL_DESC_TRANSA:
            case CUBLASLT_MATMUL_DESC_TRANSB:
            case CUBLASLT_MATMUL_DESC_TRANSC:
                return publish(output, bytes, written, CUBLAS_OP_N);
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    };
    sdk.lt.preference_get = [](cublasLtMatmulPreference_t preference,
                               cublasLtMatmulPreferenceAttributes_t attribute, void* output,
                               std::size_t bytes, std::size_t* written) {
        valid_object(preference, CudaLtObjectKind::kPreference);
        expect(attribute == CUBLASLT_MATMUL_PREF_SEARCH_MODE, "original preference query");
        touch(Phase::kPreference);
        return publish(output, bytes, written, std::uint32_t{0});
    };
    sdk.lt.heuristic = [](cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
                          cublasLtMatrixLayout_t a, cublasLtMatrixLayout_t b,
                          cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
                          cublasLtMatmulPreference_t preference, int requested,
                          cublasLtMatmulHeuristicResult_t* output, int* count) {
        valid_handle(handle);
        valid_object(operation, CudaLtObjectKind::kOperation);
        valid_object(preference, CudaLtObjectKind::kPreference);
        for (auto* layout : {a, b, c, d}) valid_object(layout, CudaLtObjectKind::kLayout);
        expect(requested == 1 && output && count, "one original algorithm result");
        touch(Phase::kHeuristic);
        *output = {};
        // The full opaque algorithm is published by a real injected query,
        // rather than fabricated in the application's admission request.
        output->algo.data[0] = 17;
        output->state = CUBLAS_STATUS_SUCCESS;
        *count = 1;
        return state.query_status;
    };
    sdk.lt.configuration_get = [](const cublasLtMatmulAlgo_t* algorithm,
                                  cublasLtMatmulAlgoConfigAttributes_t attribute, void* output,
                                  std::size_t bytes, std::size_t* written) {
        expect(algorithm && algorithm->data[0] == 17, "configuration uses original algorithm");
        touch(Phase::kConfiguration);
        expect(attribute == CUBLASLT_ALGO_CONFIG_SPLITK_NUM ||
                   attribute == CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
               "exact direct-reduction configuration query");
        return publish(output, bytes, written, std::uint32_t{0});
    };
    sdk.lt.algorithm_check =
        [](cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t a,
           cublasLtMatrixLayout_t b, cublasLtMatrixLayout_t c, cublasLtMatrixLayout_t d,
           const cublasLtMatmulAlgo_t* algorithm, cublasLtMatmulHeuristicResult_t* output) {
            valid_handle(handle);
            valid_object(operation, CudaLtObjectKind::kOperation);
            for (auto* layout : {a, b, c, d}) valid_object(layout, CudaLtObjectKind::kLayout);
            expect(algorithm && algorithm->data[0] == 17 && output, "original algorithm check");
            touch(Phase::kAlgorithmCheck);
            *output = {};
            output->algo = *algorithm;
            output->state = CUBLAS_STATUS_SUCCESS;
            return state.query_status;
        };
    sdk.lt.matmul = [](cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, const void* alpha,
                       const void* a, cublasLtMatrixLayout_t a_layout, const void* b,
                       cublasLtMatrixLayout_t b_layout, const void* beta, const void* c,
                       cublasLtMatrixLayout_t c_layout, void* d, cublasLtMatrixLayout_t d_layout,
                       const cublasLtMatmulAlgo_t* algorithm, void*, std::size_t workspace_bytes,
                       cudaStream_t stream) {
        valid_handle(handle);
        valid_object(operation, CudaLtObjectKind::kOperation);
        for (auto* layout : {a_layout, b_layout, c_layout, d_layout})
            valid_object(layout, CudaLtObjectKind::kLayout);
        expect(alpha && beta && a && b && c && d && algorithm && !workspace_bytes && !stream,
               "complete native command reaches the original callback once");
        ++state.matmuls;
        return CUBLAS_STATUS_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return sdk;
}

void driver_success(const CudaSessionResult& result) {
    expect(result.status == CUDA_SUCCESS && result.disposition == CudaSessionDisposition::kComplete,
           "owned fake Driver operation succeeds");
}
void query_success(const CudaSessionLtAlgorithms& result) {
    expect(result.result.status == CUDA_SUCCESS &&
               result.result.disposition == CudaSessionDisposition::kComplete &&
               result.result.original_status == CUBLAS_STATUS_SUCCESS &&
               result.algorithms.size() == 1 && result.sdk_version == 120604,
           "original query success and opaque algorithm survive callback exclusion");
}
void run(const Case& test, Fault fault = Fault::kNone) {
    driver_fixture::reset();
    state = {};
    state.case_name = test.name;
    CudaExecutionSession session({31, 32}, 33, dispatch());
    driver_success(session.initialize(0));
    const auto context = driver_fixture::current();
    const auto created = session.create_lt();
    expect(created.status == CUDA_SUCCESS && created.original_status == CUBLAS_STATUS_SUCCESS &&
               created.id,
           "original Lt create establishes the owned cached handle");
    CudaSessionLtQuery query;
    query.handle = created.id;
    query.operation = object(CudaLtObjectKind::kOperation);
    for (auto& layout : query.matrices) layout = object(CudaLtObjectKind::kLayout);
    query.preference = object(CudaLtObjectKind::kPreference);
    query.requested = 1;
    const auto algorithms = session.query_lt_algorithms(query);
    query_success(algorithms);
    CudaSessionLtMatmul problem;
    problem.descriptors = query;
    problem.descriptors.algorithm = algorithms.algorithms.front().algo;
    problem.scalar_bytes = 4;
    const float alpha = 1;
    std::memcpy(problem.alpha.data(), &alpha, sizeof(alpha));
    for (auto& range : problem.matrices) {
        const auto allocation = session.allocate(16);
        driver_success(allocation);
        range = {allocation.id, 0};
    }
    state.probe = [&] {
        const auto calls = state.calls;
        const auto switches = state.context_sets;
        const auto destroyed = session.destroy_lt(created.id);
        callback_expect(destroyed.status == CUDA_ERROR_NOT_READY &&
                            destroyed.disposition == CudaSessionDisposition::kDeferred &&
                            !destroyed.original_status,
                        "reentrant destroy must defer without invalidating cached Lt handle");
        const auto rejected_query = session.query_lt_algorithms(query);
        callback_expect(rejected_query.result.status == CUDA_ERROR_NOT_READY &&
                            !rejected_query.result.original_status &&
                            rejected_query.algorithms.empty(),
                        "nested query must reject before another native/context callback");
        const auto nested = session.accept_lt_matmul(1, problem);
        callback_expect(nested.status == ExecutionAcceptStatus::kInvalid && !nested.accepted(),
                        "nested admission must not consume the sequence or call native SDK");
        const auto closed = session.shutdown();
        callback_expect(closed.status == CUDA_ERROR_NOT_READY &&
                            closed.disposition == CudaSessionDisposition::kDeferred &&
                            !closed.original_status,
                        "reentrant shutdown must not close or destroy the owned context");
        callback_expect(session.activate().status == CUDA_ERROR_NOT_READY &&
                            session.deactivate().status == CUDA_ERROR_NOT_READY,
                        "reentrant activation/deactivation must not switch context");
        callback_expect(state.calls == calls && state.context_sets == switches && !state.destroys &&
                            !driver_fixture::control.context_destroys.load() &&
                            driver_fixture::current() == context && state.handle.alive &&
                            !state.matmuls,
                        "all rejected reentry leaves native counters, context and owners intact");
    };
    unsigned int native_calls_at_loss{};
    if (fault != Fault::kNone) {
        state.probe = [&] {
            native_calls_at_loss = state.native_calls;
            driver_fixture::driver.current = nullptr;
        };
        if (fault == Fault::kContextLossAndNativeError)
            state.query_status = CUBLAS_STATUS_NOT_SUPPORTED;
    }
    state.armed = test.phase;
    if (fault != Fault::kNone) {
        if (test.route == Route::kAdmission) {
            const auto rejected = session.accept_lt_matmul(1, problem);
            expect(rejected.status == ExecutionAcceptStatus::kInvalid && !rejected.accepted(),
                   "lost admission context never authorizes a queued command");
        } else {
            auto outer = query;
            if (test.route == Route::kAlgorithmCheck)
                outer.algorithm = algorithms.algorithms.front().algo;
            const auto rejected = session.query_lt_algorithms(outer);
            const bool issued =
                test.phase == Phase::kHeuristic || test.phase == Phase::kAlgorithmCheck;
            expect(rejected.result.disposition == CudaSessionDisposition::kRejected &&
                       rejected.result.status == CUDA_ERROR_INVALID_CONTEXT &&
                       rejected.algorithms.empty(),
                   "context proof failure cannot publish algorithm values");
            expect(rejected.result.original_status.has_value() == issued &&
                       (!issued || rejected.result.original_status == state.query_status),
                   "original query success/error survives proof loss; private observation has "
                   "no requested query provenance");
        }
        expect(state.probes == 1 && state.armed == Phase::kNone &&
                   state.native_calls == native_calls_at_loss && !state.matmuls &&
                   state.handle.alive && !state.destroys &&
                   !driver_fixture::control.context_destroys.load() &&
                   driver_fixture::driver.events.empty() && !driver_fixture::current(),
               "no later Lt SDK call, computation or native teardown follows callback context "
               "loss");
        const auto issued = session.issue_next();
        expect(issued.state != CudaExecutionIssueState::kIssued && !issued.library_status &&
                   state.native_calls == native_calls_at_loss && !state.matmuls,
               "rejected preflight leaves no original command available to issue");
        // This isolated child deliberately retains uncertain ownership. Process
        // exit is not presented as successful native cleanup or leak evidence.
        std::_Exit(EXIT_SUCCESS);
    }
    ExecutionAcceptance accepted;
    if (test.route == Route::kAdmission) {
        accepted = session.accept_lt_matmul(1, problem);
        expect(accepted.accepted(),
               "outer admission survives blocked callbacks and retains sequence one");
    } else {
        auto outer = query;
        if (test.route == Route::kAlgorithmCheck)
            outer.algorithm = algorithms.algorithms.front().algo;
        query_success(session.query_lt_algorithms(outer));
        accepted = session.accept_lt_matmul(1, problem);
        expect(accepted.accepted(),
               "query callbacks neither close session nor consume the first command sequence");
    }
    expect(state.probes == 1 && state.armed == Phase::kNone && !state.matmuls,
           "the selected original/context callback actually ran exactly one reentry probe");
    state.probe = {};
    const auto issued = session.issue_next();
    expect(issued.state == CudaExecutionIssueState::kIssued &&
               issued.library_status == CUBLAS_STATUS_SUCCESS && state.matmuls == 1,
           "admitted outer work issues exactly once after query/admission returns");
    expect(session.acknowledge_result(accepted.receipt), "published host result is acknowledged");
    expect(session.poll().retired == 1, "real fake event retires the accepted command");
    // A fresh call after the scope must retain exact native failure provenance.
    state.query_status = CUBLAS_STATUS_NOT_SUPPORTED;
    const auto failed = session.query_lt_algorithms(query);
    expect(failed.result.status == CUDA_SUCCESS &&
               failed.result.disposition == CudaSessionDisposition::kComplete &&
               failed.result.original_status == CUBLAS_STATUS_NOT_SUPPORTED &&
               failed.algorithms.empty(),
           "native query error is unchanged and does not poison the scoped call gate");
    state.query_status = CUBLAS_STATUS_SUCCESS;
    query_success(session.query_lt_algorithms(query));
    for (const auto& range : problem.matrices) driver_success(session.release(range.allocation));
    const auto destroyed = session.destroy_lt(created.id);
    expect(destroyed.status == CUDA_SUCCESS && destroyed.original_status == CUBLAS_STATUS_SUCCESS &&
               state.destroys == 1,
           "quiescent original Lt teardown succeeds once after the callback scopes end");
    driver_success(session.shutdown());
    expect(driver_fixture::control.context_destroys.load() == 1 &&
               driver_fixture::driver.memory.empty() && driver_fixture::driver.events.empty() &&
               driver_fixture::driver.contexts.empty(),
           "normal owner cleanup leaves no native resources");
}
void context_loss(const Case& test, Fault fault) {
    const auto child = ::fork();
    expect(child >= 0, "isolated context-loss process");
    if (!child) {
        try {
            run(test, fault);
        } catch (const std::exception& error) {
            std::cerr << "context-loss/" << test.name << ": " << error.what() << '\n';
        }
        std::_Exit(EXIT_FAILURE);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
               WEXITSTATUS(status) == EXIT_SUCCESS,
           "isolated context-loss assertions pass without unsafe native cleanup");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 2)
            throw std::runtime_error("optional argument: exact case name or --context-loss");
        std::size_t count{};
        for (const auto& test : kCases) {
            if (argc == 2 && std::string_view(argv[1]) == "--context-loss") continue;
            if (argc == 2 && test.name != argv[1]) continue;
            run(test);
            ++count;
        }
        if (argc == 1 || std::string_view(argv[1]) == "--context-loss") {
            for (const auto& test : kCases) {
                context_loss(test, Fault::kContextLoss);
                ++count;
                if (test.route != Route::kAdmission &&
                    (test.phase == Phase::kHeuristic || test.phase == Phase::kAlgorithmCheck)) {
                    context_loss(test, Fault::kContextLossAndNativeError);
                    ++count;
                }
            }
        }
        expect(count != 0, "known selected reentry case");
        std::cout << "Lt callback reentry cases passed: " << count << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << state.case_name << ": " << error.what() << '\n';
        return 1;
    }
}
