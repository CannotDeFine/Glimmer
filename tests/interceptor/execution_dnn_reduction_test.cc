#include "runtime.h"
#include "internal/process_identity.h"
#include "../backend/dnn_reduction_fixture.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <source_location>
#include <dlfcn.h>
#include <unistd.h>

// Exercise the actual deprecated ABI, not a test-only replacement signature.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" unsigned int glimmer_dnn_reduction_native_calls();
namespace {
using namespace glimmer::app;
namespace fixture = glimmer::tests::dnn_reduction;
namespace foundation = fixture::foundation;
namespace base = glimmer::tests::dnn;
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location where = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public reduction SDK status " + std::to_string(status) +
                                 " at line " + std::to_string(where.line()));
}
void check(CUresult status, std::source_location where = std::source_location::current()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("public reduction Driver status " + std::to_string(status) +
                                 " at line " + std::to_string(where.line()));
}
template <typename Function>
void symbol(void* library, const char* name, Function direct) {
    expect(::dlsym(library, name) == reinterpret_cast<void*>(direct) &&
               ::dlsym(RTLD_DEFAULT, name) == reinterpret_cast<void*>(direct),
           "direct/global/explicit lookup selects the controlled reduction ABI");
}
void lookup(void* library) {
#define CHECK(NAME) symbol(library, #NAME, &(NAME))
    CHECK(cudnnCreateReduceTensorDescriptor);
    CHECK(cudnnDestroyReduceTensorDescriptor);
    CHECK(cudnnSetReduceTensorDescriptor);
    CHECK(cudnnGetReductionIndicesSize);
    CHECK(cudnnGetReductionWorkspaceSize);
    CHECK(cudnnReduceTensor);
    CHECK(cudnnGetReduceTensorDescriptor);
#undef CHECK
}
struct Descriptors {
    cudnnTensorDescriptor_t x{}, y{};
    cudnnReduceTensorDescriptor_t reduction{};
};
cudnnStatus_t set_reduction(cudnnReduceTensorDescriptor_t descriptor, const fixture::Case& value) {
    const auto& p = value.reduction;
    return cudnnSetReduceTensorDescriptor(
        descriptor, static_cast<cudnnReduceTensorOp_t>(p.operation),
        static_cast<cudnnDataType_t>(p.type), static_cast<cudnnNanPropagation_t>(p.nan),
        static_cast<cudnnReduceTensorIndices_t>(p.indices),
        static_cast<cudnnIndicesType_t>(p.indices_type));
}
void configure(const Descriptors& d, const fixture::Case& value) {
    check(cudnnSetTensorNdDescriptor(d.x, static_cast<cudnnDataType_t>(value.x.type),
                                     value.x.dimensions, value.x.sizes.data(),
                                     value.x.strides.data()));
    check(cudnnSetTensorNdDescriptor(d.y, static_cast<cudnnDataType_t>(value.y.type),
                                     value.y.dimensions, value.y.sizes.data(),
                                     value.y.strides.data()));
    check(set_reduction(d.reduction, value));
}
Descriptors metadata() {
    Descriptors d;
    check(cudnnCreateTensorDescriptor(&d.x));
    check(cudnnCreateTensorDescriptor(&d.y));
    check(cudnnCreateReduceTensorDescriptor(&d.reduction));
    configure(d, fixture::make_case(CUDNN_REDUCE_TENSOR_ADD));
    auto operation = CUDNN_REDUCE_TENSOR_MAX;
    auto type = CUDNN_DATA_DOUBLE;
    auto nan = CUDNN_PROPAGATE_NAN;
    auto indices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
    auto indices_type = CUDNN_64BIT_INDICES;
    expect(cudnnGetReduceTensorDescriptor(d.reduction, &operation, &type, &nan, &indices,
                                          &indices_type) == CUDNN_STATUS_NOT_SUPPORTED &&
               operation == CUDNN_REDUCE_TENSOR_MAX && type == CUDNN_DATA_DOUBLE &&
               nan == CUDNN_PROPAGATE_NAN && indices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES &&
               indices_type == CUDNN_64BIT_INDICES,
           "reference descriptor getter remains a typed output-preserving guard");
    auto unsupported = fixture::make_case(CUDNN_REDUCE_TENSOR_ADD);
    unsupported.reduction.operation = 15;
    expect(set_reduction(d.reduction, unsupported) == CUDNN_STATUS_NOT_SUPPORTED,
           "representable unsupported operation reaches original metadata status");
    cudnnReduceTensorDescriptor_t stale{};
    check(cudnnCreateReduceTensorDescriptor(&stale));
    check(cudnnDestroyReduceTensorDescriptor(stale));
    expect(cudnnDestroyReduceTensorDescriptor(stale) == CUDNN_STATUS_NOT_INITIALIZED &&
               !foundation::control.context_creates.load(),
           "reduction metadata remains context-independent and rejects stale identities");
    return d;
}
struct Cached {
    decltype(&cudnnReduceTensor) compute{};
    decltype(&cudnnGetReductionIndicesSize) indices{};
    decltype(&cudnnGetReductionWorkspaceSize) workspace{};
};
enum class Buffers {
    kPositive,
    kNullIgnored,
    kZeroInterior,
    kZeroEnd,
    kShortIndices,
    kShortWorkspace,
    kNullWorkspace
};
void queries(Cached cached, cudnnHandle_t handle, const Descriptors& d, const fixture::Case& value,
             bool detailed) {
    std::size_t indices = 991, workspace = 992;
    check(cached.indices(handle, d.reduction, d.x, d.y, &indices));
    check(cached.workspace(handle, d.reduction, d.x, d.y, &workspace));
    expect(indices == (fixture::needs_indices(value.reduction) ? 16U : 0U) &&
               workspace == fixture::required_workspace.load(),
           "original queries retain separate full-width native sizes");
    if (!detailed) return;
    fixture::override_query.store(true);
    fixture::query_value.store(SIZE_MAX - 7);
    check(cached.indices(handle, d.reduction, d.x, d.y, &indices));
    check(cached.workspace(handle, d.reduction, d.x, d.y, &workspace));
    expect(indices == SIZE_MAX - 7 && workspace == SIZE_MAX - 7,
           "full native size_t survives wire and caller publication");
    fixture::override_query.store(false);
    for (const auto query : {cached.indices, cached.workspace}) {
        std::size_t sentinel = 117;
        fixture::fail_query.store(true);
        expect(
            query(handle, d.reduction, d.x, d.y, &sentinel) == CUDNN_STATUS_NOT_SUPPORTED_SHAPE &&
                sentinel == 117,
            "detailed native query failure never publishes a private touched output");
    }
}
struct Storage {
    std::array<CUdeviceptr, 4> addresses{};
    std::array<std::vector<std::byte>, 4> images;
};
Storage allocate(const fixture::Case& value) {
    Storage storage{
        {},
        {fixture::image(value, false), fixture::image(value, true),
         std::vector<std::byte>(48, std::byte{0xcd}), std::vector<std::byte>(64, std::byte{0xcd})}};
    for (std::size_t i = 0; i != storage.addresses.size(); ++i) {
        check(cuMemAlloc(&storage.addresses[i], storage.images[i].size()));
        check(
            cuMemcpyHtoD(storage.addresses[i], storage.images[i].data(), storage.images[i].size()));
    }
    return storage;
}
cudnnStatus_t invoke(Cached cached, cudnnHandle_t handle, const Descriptors& d,
                     const fixture::Case& value, const Storage& storage, Buffers mode) {
    void* indices = reinterpret_cast<void*>(storage.addresses[2] + 16);
    void* workspace = reinterpret_cast<void*>(storage.addresses[3] + 16);
    std::size_t indices_bytes = fixture::needs_indices(value.reduction) ? 16 : 0;
    std::size_t workspace_bytes = fixture::required_workspace.load();
    if (mode == Buffers::kNullIgnored) {
        indices = workspace = nullptr;
        indices_bytes = SIZE_MAX - 13;
        workspace_bytes = SIZE_MAX - 17;
    } else if (mode == Buffers::kZeroEnd) {
        indices = reinterpret_cast<void*>(storage.addresses[2] + storage.images[2].size());
        workspace = reinterpret_cast<void*>(storage.addresses[3] + storage.images[3].size());
    } else if (mode == Buffers::kShortIndices)
        indices_bytes = 15;
    else if (mode == Buffers::kShortWorkspace)
        workspace_bytes = 31;
    else if (mode == Buffers::kNullWorkspace)
        workspace = nullptr;
    const float alpha = static_cast<float>(value.alpha), beta = static_cast<float>(value.beta);
    return cached.compute(
        handle, d.reduction, indices, indices_bytes, workspace, workspace_bytes,
        value.x.type == CUDNN_DATA_DOUBLE ? static_cast<const void*>(&value.alpha) : &alpha, d.x,
        reinterpret_cast<void*>(storage.addresses[0] + 16),
        value.x.type == CUDNN_DATA_DOUBLE ? static_cast<const void*>(&value.beta) : &beta, d.y,
        reinterpret_cast<void*>(storage.addresses[1] + 16));
}
void arithmetic(Cached cached, cudnnHandle_t handle, const Descriptors& d, fixture::Case value,
                Buffers mode = Buffers::kPositive, bool fails = false, bool detailed = false) {
    fixture::required_workspace.store(
        mode == Buffers::kNullIgnored || mode == Buffers::kZeroInterior || mode == Buffers::kZeroEnd
            ? 0
            : 32);
    configure(d, value);
    queries(cached, handle, d, value, detailed);
    auto storage = allocate(value);
    fixture::fail_compute.store(fails);
    const auto status = invoke(cached, handle, d, value, storage, mode);
    const auto wanted = fails ? CUDNN_STATUS_EXECUTION_FAILED
                        : mode == Buffers::kShortIndices || mode == Buffers::kShortWorkspace ||
                                mode == Buffers::kNullWorkspace
                            ? CUDNN_STATUS_INVALID_VALUE
                            : CUDNN_STATUS_SUCCESS;
    expect(status == wanted,
           "actual SDK reduction status, including insufficient buffers, is preserved");
    expect(fixture::observed_indices_bytes.load() == (mode == Buffers::kNullIgnored ? SIZE_MAX - 13
                                                      : mode == Buffers::kShortIndices ? 15U
                                                      : fixture::needs_indices(value.reduction)
                                                          ? 16U
                                                          : 0U) &&
               fixture::observed_workspace_bytes.load() ==
                   (mode == Buffers::kNullIgnored      ? SIZE_MAX - 17
                    : mode == Buffers::kShortWorkspace ? 31U
                                                       : fixture::required_workspace.load()) &&
               fixture::observed_null_indices.load() == (mode == Buffers::kNullIgnored) &&
               fixture::observed_null_workspace.load() ==
                   (mode == Buffers::kNullIgnored || mode == Buffers::kNullWorkspace),
           "public frontend preserves original optional identity and byte counts independently");
    auto actual = storage.images;
    for (std::size_t i = 0; i != actual.size(); ++i)
        check(cuMemcpyDtoH(actual[i].data(), storage.addresses[i], actual[i].size()));
    expect(actual[0] == storage.images[0], "public source, guards and stride holes are unchanged");
    fixture::verify(value, actual[1], storage.images[1], wanted != CUDNN_STATUS_SUCCESS);
    fixture::verify_auxiliary(actual[2], actual[3], value, wanted != CUDNN_STATUS_SUCCESS);
    for (const auto address : storage.addresses) check(cuMemFree(address));
}
void pending_error(Cached cached, cudnnHandle_t handle, const Descriptors& d, CUcontext context) {
    auto value = fixture::make_case(CUDNN_REDUCE_TENSOR_MAX);
    fixture::required_workspace.store(32);
    configure(d, value);
    const auto storage = allocate(value);
    foundation::control.can_stop.store(false);
    fixture::fail_compute.store(true);
    expect(invoke(cached, handle, d, value, storage, Buffers::kPositive) ==
               CUDNN_STATUS_EXECUTION_FAILED,
           "native error returns without claiming physical completion");
    queries(cached, handle, d, value, false);
    auto mutation =
        std::async(std::launch::async, [&] { return set_reduction(d.reduction, value); });
    auto tensor_release =
        std::async(std::launch::async, [&] { return cudnnDestroyTensorDescriptor(d.y); });
    auto handle_release = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        const auto result = cudnnDestroy(handle);
        check(cuCtxSetCurrent(nullptr));
        return result;
    });
    std::array<std::future<CUresult>, 4> releases;
    for (std::size_t i = 0; i != releases.size(); ++i)
        releases[i] = std::async(std::launch::async, [&, i] {
            check(cuCtxSetCurrent(context));
            const auto result = cuMemFree(storage.addresses[i]);
            check(cuCtxSetCurrent(nullptr));
            return result;
        });
    bool pending =
        mutation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        tensor_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        handle_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    for (auto& release : releases)
        pending &= release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    check(mutation.get());
    check(tensor_release.get());
    check(handle_release.get());
    for (auto& release : releases) check(release.get());
    expect(pending, "all seven metadata/handle/allocation operations await the physical event");
}
void application(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent reduction caller provider");
    lookup(library);
    const Cached cached{
        reinterpret_cast<decltype(&cudnnReduceTensor)>(::dlsym(library, "cudnnReduceTensor")),
        reinterpret_cast<decltype(&cudnnGetReductionIndicesSize)>(
            ::dlsym(library, "cudnnGetReductionIndicesSize")),
        reinterpret_cast<decltype(&cudnnGetReductionWorkspaceSize)>(
            ::dlsym(library, "cudnnGetReductionWorkspaceSize"))};
    const auto d = metadata();
    expect(::dlclose(library) == 0, "cached reduction wrappers outlive explicit provider handle");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cudnnHandle_t handle{};
    check(cudnnCreate(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
    for (bool is_double : {false, true}) {
        if (is_double) check(cudnnSetStream(handle, nullptr));
        for (std::uint32_t op = 0; op <= CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS; ++op)
            arithmetic(cached, handle, d, fixture::make_case(op, is_double), Buffers::kPositive,
                       false, !is_double && op == 0);
    }
    for (bool input_is_int8 : {false, true}) {
        auto mixed = fixture::make_case(CUDNN_REDUCE_TENSOR_ADD);
        (input_is_int8 ? mixed.x : mixed.y).type = CUDNN_DATA_INT8;
        mixed.alpha = 2;
        mixed.beta = 1;
        arithmetic(cached, handle, d, mixed);
    }
    for (auto mode : {Buffers::kNullIgnored, Buffers::kZeroInterior, Buffers::kZeroEnd,
                      Buffers::kShortWorkspace, Buffers::kNullWorkspace})
        arithmetic(cached, handle, d, fixture::make_case(CUDNN_REDUCE_TENSOR_ADD), mode);
    arithmetic(cached, handle, d, fixture::make_case(CUDNN_REDUCE_TENSOR_MIN),
               Buffers::kShortIndices);
    arithmetic(cached, handle, d, fixture::make_case(CUDNN_REDUCE_TENSOR_ADD), Buffers::kPositive,
               true);
    pending_error(cached, handle, d, context);
    check(cudnnDestroyTensorDescriptor(d.x));
    check(cudnnDestroyReduceTensorDescriptor(d.reduction));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    expect(!glimmer_dnn_reduction_native_calls(), "no caller-native reduction function executes");
}
void run(const char* provider) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual reduction service process identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-reduction-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 256;
    config.exit_when_idle = true;
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated original-owned reduction IPC");
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start independent reduction execution service");
    client = std::async(std::launch::async, [&] { application(provider); });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready &&
           Clock::now() < deadline)
        expect(service.step(1), "progress reduction IPC");
    expect(client.wait_for(std::chrono::milliseconds{0}) == std::future_status::ready,
           "bounded reduction public pipeline");
    client.get();
    expect(service.request_stop() && service.finished(), "reduction service physically closes");
    expect(fixture::state.computes == 28 && fixture::state.creates == 2 &&
               fixture::state.destroys == 2 && base::state.destroys == 2 &&
               base::state.handle_destroys == 1 && foundation::driver.frees == 112 &&
               foundation::driver.memory.empty() && foundation::driver.streams.empty() &&
               foundation::driver.events.empty() && foundation::driver.contexts.empty() &&
               foundation::control.context_creates.load() == 2 &&
               foundation::control.context_destroys.load() == 2 &&
               foundation::control.host_registers.load() > 0 &&
               foundation::control.host_registers.load() ==
                   foundation::control.host_unregisters.load() &&
               !foundation::control.context_synchronizes.load() &&
               !glimmer_dnn_reduction_native_calls(),
           "all 28 native reductions and 112 device allocations retire exactly once without a "
           "query drain");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2, "reduction provider path required");
        run(argv[1]);
        std::cout << "public reduction CPU numerical, query, routing and lifetime checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#pragma GCC diagnostic pop
