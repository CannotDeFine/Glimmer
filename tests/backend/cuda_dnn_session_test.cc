#include "internal/cuda_dnn_execution.h"
#include "dnn_fixture.h"

#include <array>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
namespace dnn = glimmer::tests::dnn;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct DriverState {
    CUcontext current{};
    char context{};
    std::array<char, 32> events{}, streams{};
    std::map<CUdeviceptr, std::size_t> allocations;
    unsigned int event_count{}, stream_count{}, stream_destroys{}, context_destroys{}, frees{};
    bool has_completed{};
} driver;
CUcontext current() {
    return driver.current;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        *output = reinterpret_cast<CUcontext>(&driver.context);
        driver.current = *output;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext context) {
        driver.current = context;
        return CUDA_SUCCESS;
    };
    sdk.context_destroy = [](CUcontext context) {
        expect(context == driver.current, "only owner destroys the current native context");
        ++driver.context_destroys;
        driver.current = nullptr;
        return CUDA_SUCCESS;
    };
    sdk.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.module_unload = [](CUmodule) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        *output = 0x10000U * (driver.allocations.size() + 1);
        driver.allocations.emplace(*output, bytes);
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr address) {
        expect(driver.allocations.erase(address) == 1, "device allocation retires exactly once");
        ++driver.frees;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr, const void*, std::size_t) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.download = [](void*, CUdeviceptr, std::size_t) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.stream_create = [](CUstream* output, unsigned int) {
        *output = reinterpret_cast<CUstream>(&driver.streams.at(driver.stream_count++));
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream) {
        ++driver.stream_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* output) {
        *output = driver.current;
        return CUDA_SUCCESS;
    };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.execution.event_create = [](CUevent* output, unsigned int) {
        *output = reinterpret_cast<CUevent>(&driver.events.at(driver.event_count++));
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent, CUstream) { return CUDA_SUCCESS; };
    sdk.execution.event_query = [](CUevent) {
        return driver.has_completed ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent) { return CUDA_SUCCESS; };
    sdk.execution.stream_synchronize = [](CUstream) {
        driver.has_completed = true;
        return CUDA_SUCCESS;
    };
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dnn::dispatch());
    return sdk;
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "native session operation succeeds");
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "native cuDNN operation succeeds");
}
void healthy_handles() {
    driver = {};
    dnn::reset(current);
    CudaSessionLimits limits;
    limits.max_objects = 2;
    CudaExecutionSession session({31, 32}, 33, dispatch(), limits);
    success(session.initialize(0));
    dnn::state.handle_create_status = CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED;
    const auto empty_failure = session.create_dnn();
    expect(
        empty_failure.disposition == CudaDnnDisposition::kComplete &&
            empty_failure.status == CUDA_SUCCESS &&
            empty_failure.original_status == dnn::state.handle_create_status && !empty_failure.id &&
            !empty_failure.value && session.snapshot().dnn_handles == 0 &&
            !session.snapshot().is_quarantined && dnn::state.handle_creates == 1 &&
            dnn::state.handle_destroys == 0,
        "known-empty native creation failure preserves status without consuming an owned handle");
    dnn::state.handle_create_status = CUDNN_STATUS_SUCCESS;
    const auto first = session.create_dnn(), second = session.create_dnn();
    success(first);
    success(second);
    expect(first.id && second.id > first.id && session.snapshot().dnn_handles == 2,
           "two native handles receive distinct nonrecycled ownership identities");
    const auto over_bound = session.create_dnn();
    expect(over_bound.disposition == CudaDnnDisposition::kRejected &&
               over_bound.status == CUDA_ERROR_OUT_OF_MEMORY && !over_bound.original_status &&
               !over_bound.id && dnn::state.handle_creates == 3,
           "handle capacity rejects before another native creation");
    auto queried = session.get_dnn_stream(first.id);
    success(queried);
    expect(queried.value == 0, "new native handle starts on the owned default stream");
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    success(stream);
    success(session.set_dnn_stream(first.id, stream.id));
    queried = session.get_dnn_stream(first.id);
    success(queried);
    expect(queried.value == stream.id, "native bound stream is projected to its owned identity");
    dnn::state.stream_status = CUDNN_STATUS_BAD_PARAM;
    queried = session.get_dnn_stream(first.id);
    expect(queried.disposition == CudaDnnDisposition::kComplete && queried.status == CUDA_SUCCESS &&
               queried.original_status == CUDNN_STATUS_BAD_PARAM && queried.value == 0 &&
               !session.snapshot().is_quarantined,
           "failed native get-stream output stays unpublished without inventing mutation");
    dnn::state.stream_status = CUDNN_STATUS_SUCCESS;
    const auto sets = dnn::state.stream_sets;
    const auto unknown_stream = session.set_dnn_stream(first.id, stream.id + 999);
    expect(unknown_stream.status == CUDA_ERROR_INVALID_HANDLE && !unknown_stream.original_status &&
               dnn::state.stream_sets == sets,
           "an unowned stream identity cannot reach native handle configuration");
    expect(session.destroy_stream(stream.id).status == CUDA_ERROR_NOT_READY &&
               driver.stream_destroys == 0,
           "the bound cuDNN stream remains owned before explicit rebinding");
    success(session.set_dnn_stream(first.id, 0));
    queried = session.get_dnn_stream(first.id);
    success(queried);
    expect(queried.value == 0, "rebinding to default clears the old stream dependency");
    success(session.destroy_stream(stream.id));
    expect(driver.stream_destroys == 1, "rebinding permits original stream destruction");
    success(session.destroy_dnn(first.id));
    const auto gets = dnn::state.stream_gets;
    expect(session.destroy_dnn(first.id).status == CUDA_ERROR_INVALID_HANDLE &&
               session.get_dnn_stream(first.id).status == CUDA_ERROR_INVALID_HANDLE &&
               session.set_dnn_stream(first.id, 0).status == CUDA_ERROR_INVALID_HANDLE &&
               dnn::state.handle_destroys == 1 && dnn::state.stream_gets == gets,
           "stale handle identities cannot query, reconfigure or destroy a native handle twice");
    const auto third = session.create_dnn();
    success(third);
    expect(third.id > second.id && session.snapshot().dnn_handles == 2,
           "reusing a freed native address cannot revive a stale public handle");
    success(session.destroy_dnn(second.id));
    success(session.shutdown());
    success(session.shutdown());
    expect(dnn::state.handle_creates == 4 && dnn::state.handle_destroys == 3 &&
               driver.context_destroys == 1 && driver.stream_destroys == 1 &&
               session.snapshot().dnn_handles == 0,
           "explicit and implicit handle retirement each execute once before context destruction");
}
void owned_computation(bool use_double, bool add_tensor, bool fails) {
    driver = {};
    dnn::reset(current);
    const auto sdk = dispatch();
    CudaDnnRegistry metadata(*sdk.dnn);
    const auto x = metadata.create(CudaDnnObjectKind::kTensor);
    const auto y = metadata.create(CudaDnnObjectKind::kTensor);
    const auto activation = metadata.create(CudaDnnObjectKind::kActivation);
    success(x);
    success(y);
    success(activation);
    CudaDnnTensor tensor;
    tensor.type = use_double ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    tensor.dimensions = 4;
    tensor.sizes = {1, 1, 2, 3};
    tensor.strides = {6, 6, 3, 1};
    success(metadata.set_tensor(x.id, CudaDnnTensorSetter::kNd, tensor));
    success(metadata.set_tensor(y.id, CudaDnnTensorSetter::kNd, tensor));
    success(metadata.set_activation(activation.id, {1, 0, 0.0}));
    CudaSessionLimits limits;
    limits.max_objects = 2;
    CudaExecutionSession session({11, 12}, 13, sdk, limits);
    expect(!session.create_dnn().original_status && dnn::state.handle_creates == 0,
           "handle creation cannot run before owned context");
    success(session.initialize(0));
    const auto handle = session.create_dnn();
    success(handle);
    expect(handle.id && session.snapshot().dnn_handles == 1, "native handle has an owned identity");
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    success(stream);
    success(session.set_dnn_stream(handle.id, stream.id));
    expect(session.get_dnn_stream(handle.id).value == stream.id, "stream query returns owned ID");
    expect(session.destroy_stream(stream.id).status == CUDA_ERROR_NOT_READY,
           "live cuDNN handle pins its stream configuration");
    const auto source = session.allocate(64);
    const auto destination = session.allocate(64);
    success(source);
    success(destination);
    CudaSessionDnnCompute problem;
    problem.handle = handle.id;
    problem.kind =
        add_tensor ? CudaDnnComputeKind::kAddTensor : CudaDnnComputeKind::kActivationForward;
    if (!add_tensor)
        problem.activation = *metadata.borrow(activation.id, CudaDnnObjectKind::kActivation);
    problem.tensors = {*metadata.borrow(x.id, CudaDnnObjectKind::kTensor),
                       *metadata.borrow(y.id, CudaDnnObjectKind::kTensor)};
    for (const auto& view : problem.tensors)
        dnn::object(view.native, CudaDnnObjectKind::kTensor).bytes =
            6 * (use_double ? sizeof(double) : sizeof(float));
    problem.buffers = {{{source.id, 8}, {destination.id, 16}}};
    problem.scalar_bytes = use_double ? sizeof(double) : sizeof(float);
    const double alpha = 1.25, beta = -0.5;
    if (use_double) {
        std::memcpy(problem.alpha.data(), &alpha, sizeof(alpha));
        std::memcpy(problem.beta.data(), &beta, sizeof(beta));
    } else {
        const float a = static_cast<float>(alpha), b = static_cast<float>(beta);
        std::memcpy(problem.alpha.data(), &a, sizeof(a));
        std::memcpy(problem.beta.data(), &b, sizeof(b));
    }
    const auto foreign = std::async(std::launch::async, [&] { return session.create_dnn(); }).get();
    expect(!foreign.original_status && dnn::state.handle_creates == 1,
           "foreign thread cannot issue handle creation");
    auto bad = problem;
    bad.buffers[1].offset = 63;
    expect(!session.accept_dnn_compute(1, bad).acceptance.accepted(),
           "actual native tensor extent cannot exceed owned allocation");
    bad = problem;
    bad.scalar_bytes = use_double ? sizeof(float) : sizeof(double);
    expect(!session.accept_dnn_compute(1, bad).acceptance.accepted(),
           "coefficient width must match native tensor type");
    bad = problem;
    bad.tensors[0].lifetime.reset();
    expect(!session.accept_dnn_compute(1, bad).acceptance.accepted(),
           "native descriptor pointer alone cannot authorize a physical consumer");
    bad = {};
    dnn::state.get_status = CUDNN_STATUS_BAD_PARAM;
    const auto query_error = session.accept_dnn_compute(1, problem);
    expect(!query_error.acceptance.accepted() &&
               query_error.original_status == CUDNN_STATUS_BAD_PARAM &&
               session.snapshot().submissions.outstanding == 0,
           "native descriptor query error is preserved without admitting work");
    dnn::state.get_status = CUDNN_STATUS_SUCCESS;
    dnn::state.size_status = CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT;
    const auto size_error = session.accept_dnn_compute(1, problem);
    expect(!size_error.acceptance.accepted() &&
               size_error.original_status == CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT &&
               session.snapshot().submissions.outstanding == 0 &&
               !session.snapshot().is_quarantined,
           "native tensor-size error remains original and admits no physical work");
    dnn::state.size_status = CUDNN_STATUS_SUCCESS;
    const auto accepted = session.accept_dnn_compute(1, problem);
    expect(accepted.acceptance.accepted() && !accepted.original_status &&
               dnn::state.activation_calls == 0 && dnn::state.add_calls == 0,
           "acceptance snapshots native arguments without executing vendor computation");
    problem = {};
    expect(metadata.destroy(x.id).disposition == CudaDnnDisposition::kDeferred &&
               metadata.set_tensor(y.id, CudaDnnTensorSetter::kNd, tensor).disposition ==
                   CudaDnnDisposition::kDeferred &&
               session.destroy_dnn(handle.id).disposition == CudaDnnDisposition::kDeferred,
           "queued compute retains immutable descriptors and context-owned handle");
    dnn::state.compute_status = fails ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_SUCCESS;
    const auto issued = session.issue_next();
    expect(issued.state == CudaExecutionIssueState::kIssued &&
               dnn::state.activation_calls == (add_tensor ? 0U : 1U) &&
               dnn::state.add_calls == (add_tensor ? 1U : 0U),
           "scheduler invokes the selected original cuDNN computation exactly once");
    const auto host = session.host_result(accepted.acceptance.receipt);
    expect(issued.native_status == CUDA_SUCCESS &&
               issued.library_status == dnn::state.compute_status && host &&
               host->status == dnn::state.compute_status &&
               host->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk,
           "original vendor error remains distinct from Driver completion and host ACK");
    const auto& invocation = dnn::state.last_call;
    expect(invocation.alpha == alpha && invocation.beta == beta &&
               invocation.double_scalars == use_double &&
               reinterpret_cast<CUdeviceptr>(invocation.x) == source.address + 8 &&
               reinterpret_cast<CUdeviceptr>(invocation.y) == destination.address + 16,
           "original computation receives exact aligned coefficients and owned address offsets");
    expect(session.acknowledge_result(accepted.acceptance.receipt), "acknowledge native result");
    expect(session.poll().retired == 0 &&
               metadata.destroy(y.id).disposition == CudaDnnDisposition::kDeferred &&
               session.release(source.id).disposition == CudaSessionDisposition::kDeferred &&
               session.set_dnn_stream(handle.id, 0).disposition == CudaDnnDisposition::kDeferred &&
               session.shutdown().disposition == CudaSessionDisposition::kDeferred,
           "host acknowledgement is not physical completion");
    driver.has_completed = true;
    expect(session.poll().retired == 1, "real trailing event retires descriptor borrows");
    success(session.release(source.id));
    success(session.release(destination.id));
    success(metadata.shutdown());
    success(session.shutdown());
    expect(dnn::state.handle_destroys == 1 && driver.context_destroys == 1 && driver.frees == 2 &&
               session.snapshot().dnn_handles == 0 && metadata.snapshot().is_closed,
           "resources retire once after physical completion");
}
void uncertain_handles() {
    for (unsigned int mode = 0; mode < 7; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "isolated uncertain handle control");
        if (!child) {
            // Ambiguous ownership intentionally survives until process exit;
            // neither failed assertions nor fake resets may trigger cleanup.
            try {
                driver = {};
                dnn::reset(current);
                auto* session = new CudaExecutionSession({21, 22}, 23, dispatch());
                success(session->initialize(0));
                if (mode == 0) dnn::state.null_handle_create = true;
                if (mode == 1) {
                    dnn::state.partial_handle_create = true;
                    dnn::state.handle_create_status = CUDNN_STATUS_INTERNAL_ERROR;
                }
                const auto created = session->create_dnn();
                if (mode <= 1) {
                    expect(
                        !created.id && created.original_status == dnn::state.handle_create_status,
                        "unproved creation retains original status without publishing ID");
                } else if (mode == 2 || mode == 3) {
                    success(created);
                    dnn::state.duplicate_handle_create = true;
                    if (mode == 3) {
                        dnn::state.partial_handle_create = true;
                        dnn::state.handle_create_status = CUDNN_STATUS_INTERNAL_ERROR;
                    }
                    expect(!session->create_dnn().id,
                           "duplicate native handle cannot have two owners");
                } else if (mode == 4) {
                    success(created);
                    dnn::state.handle_destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
                    expect(session->destroy_dnn(created.id).original_status ==
                               CUDNN_STATUS_INTERNAL_ERROR,
                           "failed destruction preserves original status");
                } else if (mode == 5) {
                    success(created);
                    dnn::state.stream_status = CUDNN_STATUS_BAD_PARAM;
                    const auto failed = session->set_dnn_stream(created.id, 0);
                    expect(failed.disposition == CudaDnnDisposition::kComplete &&
                               failed.status == CUDA_SUCCESS &&
                               failed.original_status == CUDNN_STATUS_BAD_PARAM &&
                               dnn::state.stream_sets == 1,
                           "failed original stream mutation preserves native status before "
                           "quarantine");
                } else {
                    success(created);
                    dnn::state.handles[0].stream =
                        reinterpret_cast<cudaStream_t>(&driver.streams[0]);
                    const auto mismatch = session->get_dnn_stream(created.id);
                    expect(mismatch.disposition == CudaDnnDisposition::kRejected &&
                               mismatch.status == CUDA_ERROR_UNKNOWN &&
                               mismatch.original_status == CUDNN_STATUS_SUCCESS &&
                               mismatch.value == 0,
                           "a successful native query cannot publish an unowned stream");
                }
                if (mode == 6) {
                    // Match contradictory_blas_metadata_is_not_resource_authority:
                    // a bad read-only query cannot authorize new work, but it
                    // does not erase the already proven handle/context owner.
                    expect(session->snapshot().is_quarantined &&
                               session->accept_dnn_compute(1, {}).acceptance.status ==
                                   glimmer::backend::ExecutionAcceptStatus::kQuarantined &&
                               dnn::state.activation_calls == 0 && dnn::state.add_calls == 0,
                           "unowned stream output quarantines further work without execution");
                    success(session->shutdown());
                    success(session->shutdown());
                    expect(dnn::state.handle_destroys == 1 && driver.context_destroys == 1 &&
                               driver.stream_destroys == 0 && driver.frees == 0,
                           "proven handle/context cleans once and never destroys queried foreign "
                           "stream");
                } else
                    expect(session->snapshot().is_quarantined &&
                               session->shutdown().status == CUDA_ERROR_UNKNOWN &&
                               session->shutdown().status == CUDA_ERROR_UNKNOWN &&
                               dnn::state.handle_destroys == (mode == 4 ? 1U : 0U) &&
                               driver.context_destroys == 0,
                           "uncertain native ownership cannot be retried or erased by context "
                           "teardown");
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "uncertain_handles mode " << mode << ": " << error.what() << '\n';
                ::_exit(1);
            } catch (...) {
                std::cerr << "uncertain_handles mode " << mode << ": unknown exception\n";
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated cuDNN handle ownership control passes");
    }
}
void corrupt_native_metadata() {
    // Two malformed successful outputs and two context-change outcomes for
    // each descriptor query, followed by stream-mismatch/context-loss/original-
    // rejection controls through the actual compute acceptance entry point.
    for (unsigned int mode = 0; mode < 9; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "isolated corrupt native metadata control");
        if (!child) {
            try {
                driver = {};
                dnn::reset(current);
                auto native_dnn = dnn::dispatch();
                if (mode == 2 || mode == 3)
                    native_dnn.tensor_get_nd = [](cudnnTensorDescriptor_t native, int requested,
                                                  cudnnDataType_t* type, int* dimensions,
                                                  int* sizes, int* strides) {
                        const auto status =
                            dnn::get_tensor(native, requested, type, dimensions, sizes, strides);
                        driver.current = nullptr;
                        return status;
                    };
                if (mode == 4 || mode == 5)
                    native_dnn.tensor_size = [](cudnnTensorDescriptor_t native,
                                                std::size_t* bytes) {
                        const auto status = dnn::dispatch().tensor_size(native, bytes);
                        driver.current = nullptr;
                        return status;
                    };
                if (mode == 7)
                    native_dnn.get_stream = [](cudnnHandle_t handle, cudaStream_t* stream) {
                        const auto status = dnn::dispatch().get_stream(handle, stream);
                        // The initial successful create-time query is valid.
                        // Only the later injected error changes the context.
                        if (status != CUDNN_STATUS_SUCCESS) driver.current = nullptr;
                        return status;
                    };
                auto sdk = dispatch();
                sdk.dnn = std::make_shared<const CudaDnnDispatch>(native_dnn);
                // Deliberately retain quarantined owners until child exit.
                // No destructor may try native cleanup under a changed context.
                auto* session = new CudaExecutionSession({41, 42}, 43, sdk);
                auto* metadata = new CudaDnnRegistry(dnn::dispatch());
                success(session->initialize(0));
                const auto handle = session->create_dnn();
                const auto descriptor = metadata->create(CudaDnnObjectKind::kTensor);
                const auto source = session->allocate(4), destination = session->allocate(4);
                success(handle);
                success(descriptor);
                success(source);
                success(destination);
                const CudaDnnTensor tensor{.type = CUDNN_DATA_FLOAT,
                                           .dimensions = 4,
                                           .sizes = {1, 1, 1, 1},
                                           .strides = {1, 1, 1, 1}};
                success(metadata->set_tensor(descriptor.id, CudaDnnTensorSetter::kNd, tensor));
                const auto view = metadata->borrow(descriptor.id, CudaDnnObjectKind::kTensor);
                expect(view.has_value(), "metadata corruption starts from a live owned descriptor");
                dnn::object(view->native, CudaDnnObjectKind::kTensor).bytes = sizeof(float);
                CudaSessionDnnCompute problem;
                problem.kind = CudaDnnComputeKind::kAddTensor;
                problem.handle = handle.id;
                problem.tensors = {*view, *view};
                problem.buffers = {{{source.id, 0}, {destination.id, 0}}};
                problem.scalar_bytes = sizeof(float);
                const float alpha = 1.0F;
                std::memcpy(problem.alpha.data(), &alpha, sizeof(alpha));
                if (mode == 0) dnn::state.reported_rank = CUDNN_DIM_MAX + 1;
                if (mode == 1) dnn::state.reported_type = std::numeric_limits<std::uint32_t>::max();
                if (mode == 3) dnn::state.get_status = CUDNN_STATUS_BAD_PARAM;
                if (mode == 5) dnn::state.size_status = CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT;
                if (mode == 6)
                    dnn::state.handles[0].stream =
                        reinterpret_cast<cudaStream_t>(&driver.streams[0]);
                if (mode >= 7) dnn::state.stream_status = CUDNN_STATUS_BAD_PARAM;
                const auto result = session->accept_dnn_compute(1, problem);
                const auto expected_status = mode == 3 || mode >= 7 ? CUDNN_STATUS_BAD_PARAM
                                             : mode == 5 ? CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT
                                                         : CUDNN_STATUS_SUCCESS;
                const auto expected_acceptance =
                    mode == 8 ? glimmer::backend::ExecutionAcceptStatus::kInvalid
                              : glimmer::backend::ExecutionAcceptStatus::kQuarantined;
                expect(!result.acceptance.accepted() &&
                           result.acceptance.status == expected_acceptance &&
                           result.original_status == expected_status &&
                           session->snapshot().is_quarantined == (mode != 8) &&
                           session->snapshot().submissions.outstanding == 0,
                       "native query outcome retains original status and exact "
                       "admission/quarantine class");
                if (mode == 8) {
                    dnn::state.stream_status = CUDNN_STATUS_SUCCESS;
                    success(session->get_dnn_stream(handle.id));
                    expect(
                        !session->snapshot().is_quarantined,
                        "plain native stream-query rejection does not invalidate proven ownership");
                } else
                    expect(session->accept_dnn_compute(2, problem).acceptance.status ==
                               glimmer::backend::ExecutionAcceptStatus::kQuarantined,
                           "corrupt query cannot admit a later physical consumer");
                if (mode <= 1 || mode == 6 || mode == 8) {
                    // As in the established BLAS metadata contract, malformed
                    // read-only tensor output rejects admission, not the proven
                    // resource identity. No compute was admitted or launched.
                    success(session->shutdown());
                    success(session->shutdown());
                    expect(
                        dnn::state.handle_destroys == 1 && driver.context_destroys == 1,
                        "quiescent proven handle/context cleans exactly once after bad metadata");
                } else {
                    // A lost current context is a different failure class:
                    // preserve the actual Driver gate error and do no cleanup.
                    expect(session->shutdown().status == CUDA_ERROR_INVALID_CONTEXT &&
                               session->shutdown().status == CUDA_ERROR_INVALID_CONTEXT &&
                               dnn::state.handle_destroys == 0 && driver.context_destroys == 0,
                           "changed native context blocks cleanup with the original context error");
                }
                expect(dnn::state.activation_calls == 0 && dnn::state.add_calls == 0 &&
                           dnn::state.destroys == 0 && driver.stream_destroys == 0 &&
                           driver.frees == 0,
                       "metadata rejection never executes compute or releases descriptor/foreign "
                       "stream authority");
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "corrupt_native_metadata mode " << mode << ": " << error.what()
                          << '\n';
                ::_exit(1);
            } catch (...) {
                std::cerr << "corrupt_native_metadata mode " << mode << ": unknown exception\n";
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated direct-session metadata corruption control passes");
    }
}
}  // namespace
int main() {
    try {
        healthy_handles();
        for (const auto use_double : {false, true})
            for (const auto add_tensor : {false, true})
                for (const auto fails : {false, true})
                    owned_computation(use_double, add_tensor, fails);
        uncertain_handles();
        corrupt_native_metadata();
        std::cout << "cuDNN owned computation checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
