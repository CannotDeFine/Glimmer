#include "runtime.h"
#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"
#include "../backend/context_control_fixture.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <future>
#include <memory>
#include <source_location>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_lt_native_calls();
extern "C" unsigned int glimmer_random_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::backend::cuda;
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cublasStatus_t status) {
    expect(status == CUBLAS_STATUS_SUCCESS, "owned Lt success");
}
struct Object {
    bool is_live{};
    CudaLtObjectKind kind{};
    std::array<std::array<std::byte, 8>, 32> attributes{};
};
struct Native {
    std::array<Object, 8> objects{};
    unsigned int creates{}, destroys{}, gets{}, sets{}, gpu_calls{};
    bool is_quarantine{};
} native;
template <typename Value>
void store(Object& object, unsigned int attribute, Value value) {
    std::memcpy(object.attributes.at(attribute).data(), &value, sizeof(value));
}
Object* create(CudaLtObjectKind kind) {
    const auto found =
        std::ranges::find_if(native.objects, [](const auto& object) { return !object.is_live; });
    expect(found != native.objects.end(), "bounded original descriptor storage");
    auto& object = *found;
    expect(!object.is_live, "native slot only reused after actual destruction");
    object = {};
    object.is_live = true;
    object.kind = kind;
    return &object;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): actual original SDK callbacks.
cublasStatus_t CUBLASWINAPI layout_create(cublasLtMatrixLayout_t* output, cudaDataType type,
                                          std::uint64_t rows, std::uint64_t columns,
                                          std::int64_t leading) {
    if (++native.creates == 1) return CUBLAS_STATUS_ALLOC_FAILED;
    auto* object = create(CudaLtObjectKind::kLayout);
    store(*object, 0, static_cast<std::uint32_t>(type));
    store(*object, 1, std::uint32_t{0});
    store(*object, 2, rows);
    store(*object, 3, columns);
    store(*object, 4, leading);
    store(*object, 5, std::uint32_t{1});
    *output = reinterpret_cast<cublasLtMatrixLayout_t>(object);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI operation_create(cublasLtMatmulDesc_t* output,
                                             cublasComputeType_t compute, cudaDataType scale) {
    ++native.creates;
    auto* object = create(CudaLtObjectKind::kOperation);
    store(*object, 0, static_cast<std::uint32_t>(compute));
    store(*object, 1, static_cast<std::uint32_t>(scale));
    store(*object, CUBLASLT_MATMUL_DESC_FILL_MODE, std::uint32_t{2});
    store(*object, CUBLASLT_MATMUL_DESC_EPILOGUE, std::uint32_t{1});
    store(*object, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, std::int32_t{-1});
    *output = reinterpret_cast<cublasLtMatmulDesc_t>(object);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CUBLASWINAPI preference_create(cublasLtMatmulPreference_t* output) {
    ++native.creates;
    *output = reinterpret_cast<cublasLtMatmulPreference_t>(create(CudaLtObjectKind::kPreference));
    return CUBLAS_STATUS_SUCCESS;
}
template <typename Handle>
cublasStatus_t CUBLASWINAPI destroy(Handle handle) {
    auto* object = reinterpret_cast<Object*>(handle);
    expect(object && object->is_live, "destroy original live Lt object exactly once");
    object->is_live = false;
    ++native.destroys;
    return CUBLAS_STATUS_SUCCESS;
}
template <typename Handle, typename Attribute>
cublasStatus_t CUBLASWINAPI set(Handle handle, Attribute attribute, const void* buffer,
                                std::size_t bytes) {
    auto* object = reinterpret_cast<Object*>(handle);
    const auto schema = cuda_lt_attribute(object->kind, static_cast<std::uint32_t>(attribute));
    expect(object->is_live && schema && schema->bytes == bytes, "classified original attribute");
    ++native.sets;
    if (object->kind == CudaLtObjectKind::kLayout && static_cast<unsigned int>(attribute) == 1) {
        std::uint32_t value{};
        std::memcpy(&value, buffer, sizeof(value));
        if (value > 4) return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::memcpy(object->attributes.at(static_cast<unsigned int>(attribute)).data(), buffer, bytes);
    return CUBLAS_STATUS_SUCCESS;
}
template <typename Handle, typename Attribute>
cublasStatus_t CUBLASWINAPI get(Handle handle, Attribute attribute, void* buffer, std::size_t bytes,
                                std::size_t* written) {
    auto* object = reinterpret_cast<Object*>(handle);
    const auto schema = cuda_lt_attribute(object->kind, static_cast<std::uint32_t>(attribute));
    expect(object->is_live && schema, "query actual SDK-owned metadata");
    ++native.gets;
    if (bytes && bytes != schema->bytes) return CUBLAS_STATUS_INVALID_VALUE;
    *written = native.is_quarantine ? schema->bytes + 1 : schema->bytes;
    if (bytes)
        std::memcpy(buffer, object->attributes.at(static_cast<unsigned int>(attribute)).data(),
                    bytes);
    return CUBLAS_STATUS_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
std::size_t CUBLASWINAPI version() {
    return 120604;
}
std::size_t CUBLASWINAPI runtime_version() {
    return 12031;
}
CUresult CUDAAPI initialize(unsigned int) {
    ++native.gpu_calls;
    return CUDA_ERROR_UNKNOWN;
}
CUresult CUDAAPI context_set(CUcontext) {
    ++native.gpu_calls;
    return CUDA_ERROR_UNKNOWN;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result{};
    result.initialize = initialize;
    result.context_set_current = context_set;
    auto& lt = result.lt;
    lt.layout_create = layout_create;
    lt.layout_destroy = destroy<cublasLtMatrixLayout_t>;
    lt.layout_set = set<cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t>;
    lt.layout_get = get<cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t>;
    lt.operation_create = operation_create;
    lt.operation_destroy = destroy<cublasLtMatmulDesc_t>;
    lt.operation_set = set<cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t>;
    lt.operation_get = get<cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t>;
    lt.preference_create = preference_create;
    lt.preference_destroy = destroy<cublasLtMatmulPreference_t>;
    lt.preference_set = set<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t>;
    lt.preference_get = get<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t>;
    lt.version = version;
    lt.runtime_version = runtime_version;
    return result;
}
struct ChainNative {
    CUcontext handle_context{};
    char handle{};
    std::atomic<bool> can_stop{};
    unsigned int heuristics{}, checks{}, matmuls{}, handle_destroys{}, event_destroys{};
    std::atomic<int> runtime_version{12060};
    std::atomic<unsigned int> metadata_calls{};
    unsigned int stream_creates{}, stream_releases{};
} chain;
struct RandomNative {
    std::array<char, 4> handles{};
    std::array<bool, 4> live{};
    unsigned int creates{}, destroys{}, seeds{};
    std::uint64_t seed{};
    bool quarantine{};
} random_native;
thread_local CUcontext chain_current{};
thread_local char chain_context;
thread_local std::vector<CUstream> chain_live_streams;
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected SDK callbacks.
CudaSessionDispatch chain_dispatch() {
    auto result = dispatch();
    result.runtime.version = [](int* output) {
        *output = chain.runtime_version.load();
        return cudaSuccess;
    };
    result.runtime.properties = [](cudaDeviceProp* output, int device) {
        expect(chain_current != nullptr, "Runtime properties execute on their actual SDK owner");
        ++chain.metadata_calls;
        output->multiProcessorCount = device < 0 ? 19 : 80;
        if (device < 0) return cudaErrorInvalidDevice;
        constexpr char name[] = "Owned device metadata";
        std::memcpy(output->name, name, sizeof(name));
        output->totalGlobalMem = std::size_t{16} * 1024 * 1024 * 1024;
        output->major = 7;
        return cudaSuccess;
    };
    result.runtime.attribute = [](int* output, cudaDeviceAttr attribute, int) {
        expect(chain_current != nullptr, "Runtime attribute stays on owned SDK context");
        ++chain.metadata_calls;
        if (attribute != cudaDevAttrMultiProcessorCount) return cudaErrorCallRequiresNewerDriver;
        *output = 80;
        return cudaSuccess;
    };
    result.runtime.peer_attribute = [](int* output, cudaDeviceP2PAttr, int, int) {
        expect(chain_current != nullptr, "Runtime peer attribute stays on owned SDK context");
        ++chain.metadata_calls;
        *output = 1;
        return cudaSuccess;
    };
    result.runtime.stream_priority_range = [](int* least, int* greatest) {
        expect(chain_current != nullptr, "stream priority metadata executes on SDK owner");
        ++chain.metadata_calls;
        *least = 0;
        *greatest = -2;
        return cudaSuccess;
    };
    result.runtime.peer_access = [](int* output, int device, int peer) {
        expect(chain_current != nullptr, "peer metadata executes on SDK owner");
        ++chain.metadata_calls;
        *output = device == peer ? 0 : 1;
        return cudaSuccess;
    };
    result.runtime.memory_info = [](std::size_t* available, std::size_t* total) {
        expect(chain_current != nullptr, "memory capacity executes on actual application context");
        ++chain.metadata_calls;
        *available = 1234567;
        *total = 7654321;
        return cudaSuccess;
    };
    result.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    result.device_get = [](CUdevice* value, int ordinal) {
        *value = ordinal;
        return CUDA_SUCCESS;
    };
    result.context_create = [](CUcontext* value, unsigned int, CUdevice) {
        *value = chain_current = reinterpret_cast<CUcontext>(&chain_context);
        return CUDA_SUCCESS;
    };
    result.context_set_current = [](CUcontext value) {
        chain_current = value;
        return CUDA_SUCCESS;
    };
    result.context_destroy = [](CUcontext value) {
        expect(
            value == chain_current && (value != chain.handle_context || chain.handle_destroys == 1),
            "Lt cleanup precedes its native context");
        for (auto stream : chain_live_streams) {
            delete reinterpret_cast<int*>(stream);
            ++chain.stream_releases;
        }
        chain_live_streams.clear();
        chain_current = nullptr;
        return CUDA_SUCCESS;
    };
    result.execution.context_get_current = [](CUcontext* value) {
        *value = chain_current;
        return CUDA_SUCCESS;
    };
    result.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                        unsigned int, unsigned int, unsigned int, unsigned int,
                                        CUstream, void**,
                                        void**) { return CUDA_ERROR_INVALID_VALUE; };
    result.execution.event_create = [](CUevent* value, unsigned int) {
        *value = reinterpret_cast<CUevent>(new int{0});
        return CUDA_SUCCESS;
    };
    result.execution.event_record = [](CUevent value, CUstream) {
        *reinterpret_cast<int*>(value) = chain.matmuls ? 1 : 0;
        return CUDA_SUCCESS;
    };
    result.execution.event_query = [](CUevent value) {
        return !*reinterpret_cast<int*>(value) || chain.can_stop.load() ? CUDA_SUCCESS
                                                                        : CUDA_ERROR_NOT_READY;
    };
    result.execution.event_destroy = [](CUevent value) {
        delete reinterpret_cast<int*>(value);
        ++chain.event_destroys;
        return CUDA_SUCCESS;
    };
    result.execution.stream_synchronize = [](CUstream) { return CUDA_ERROR_INVALID_VALUE; };
    result.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.module_unload = [](CUmodule) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.function_get = [](CUfunction*, CUmodule, const char*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.allocate = [](CUdeviceptr* value, std::size_t bytes) {
        auto* pointer = new std::byte[bytes]{};
        const std::array<float, 8> contents{1, 2, 3, 4, 5, 6, 7, 8};
        expect(bytes >= sizeof(contents), "bounded fake GPU backing");
        std::memcpy(pointer, contents.data(), sizeof(contents));
        *value = reinterpret_cast<CUdeviceptr>(pointer);
        return CUDA_SUCCESS;
    };
    result.release = [](CUdeviceptr value) {
        delete[] reinterpret_cast<std::byte*>(value);
        return CUDA_SUCCESS;
    };
    result.upload = [](CUdeviceptr value, const void* source, std::size_t bytes) {
        std::memcpy(reinterpret_cast<void*>(value), source, bytes);
        return CUDA_SUCCESS;
    };
    result.download = [](void* destination, CUdeviceptr value, std::size_t bytes) {
        std::memcpy(destination, reinterpret_cast<const void*>(value), bytes);
        return CUDA_SUCCESS;
    };
    result.host_register = [](void*, std::size_t, unsigned int) { return CUDA_SUCCESS; };
    result.host_unregister = [](void*) { return CUDA_SUCCESS; };
    result.host_map = [](CUdeviceptr* value, void* pointer, unsigned int) {
        *value = reinterpret_cast<CUdeviceptr>(pointer);
        return CUDA_SUCCESS;
    };
    result.upload_async = [](CUdeviceptr destination, const void* source, std::size_t bytes,
                             CUstream) {
        std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
        return CUDA_SUCCESS;
    };
    result.download_async = [](void* destination, CUdeviceptr source, std::size_t bytes, CUstream) {
        std::memcpy(destination, reinterpret_cast<const void*>(source), bytes);
        return CUDA_SUCCESS;
    };
    result.stream_create = [](CUstream* output, unsigned int flags) {
        expect(chain_current != nullptr && flags == CU_STREAM_DEFAULT,
               "per-thread default proxy uses original blocking-stream factory");
        ++chain.stream_creates;
        *output = reinterpret_cast<CUstream>(new int{0});
        chain_live_streams.push_back(*output);
        return CUDA_SUCCESS;
    };
    result.stream_create_priority = [](CUstream* output, unsigned int flags, int priority) {
        expect(chain_current != nullptr && flags == CU_STREAM_NON_BLOCKING,
               "original priority stream factory owns current context and flags");
        ++chain.stream_creates;
        *output = reinterpret_cast<CUstream>(new int{std::clamp(priority, -2, 0)});
        chain_live_streams.push_back(*output);
        return CUDA_SUCCESS;
    };
    result.stream_get_priority = [](CUstream stream, int* output) {
        expect(chain_current != nullptr, "clamped stream priority queried on actual owner");
        *output = *reinterpret_cast<int*>(stream);
        return CUDA_SUCCESS;
    };
    result.stream_destroy = [](CUstream stream) {
        expect(chain_current != nullptr, "stream cleanup stays on actual owner");
        ++chain.stream_releases;
        expect(std::erase(chain_live_streams, stream) == 1,
               "destroy a single known original stream");
        delete reinterpret_cast<int*>(stream);
        return CUDA_SUCCESS;
    };
    result.lt.create = [](cublasLtHandle_t* output) {
        chain.handle_context = chain_current;
        *output = reinterpret_cast<cublasLtHandle_t>(&chain.handle);
        return CUBLAS_STATUS_SUCCESS;
    };
    result.blas.create = [](cublasHandle_t* output) {
        chain.handle_context = chain_current;
        *output = reinterpret_cast<cublasHandle_t>(&chain.handle);
        return CUBLAS_STATUS_SUCCESS;
    };
    result.blas.destroy = [](cublasHandle_t handle) {
        expect(chain_current == chain.handle_context &&
                   handle == reinterpret_cast<cublasHandle_t>(&chain.handle),
               "cuBLAS-as-Lt alias has exactly one original cuBLAS owner");
        ++chain.handle_destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.blas.set_stream = [](cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; };
    result.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                           const float*, const float*, int, const float*, int, const float*, float*,
                           int) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    result.lt.destroy = [](cublasLtHandle_t value) {
        expect(value == reinterpret_cast<cublasLtHandle_t>(&chain.handle) &&
                   chain_current == chain.handle_context,
               "destroy only original server-owned Lt handle");
        ++chain.handle_destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.heuristic = [](cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
                             cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                             cublasLtMatmulPreference_t, int count,
                             cublasLtMatmulHeuristicResult_t* output, int* written) {
        if (++chain.heuristics == 1) return CUBLAS_STATUS_NOT_SUPPORTED;
        expect(count == 4 && chain_current == chain.handle_context,
               "original heuristic uses owned GPU context and original request count");
        output[0].algo.data[0] = 77;
        output[0].workspaceSize = 64;
        output[0].state = CUBLAS_STATUS_SUCCESS;
        output[0].wavesCount = 2.5F;
        output[0].reserved[0] = -7;
        *written = 1;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.algorithm_check = [](cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
                                   cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                   cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t* algorithm,
                                   cublasLtMatmulHeuristicResult_t* output) {
        expect(algorithm->data[0] == 77 && chain_current == chain.handle_context,
               "only original version-bound algorithm reaches SDK check");
        ++chain.checks;
        output->workspaceSize = 64;
        output->state = CUBLAS_STATUS_SUCCESS;
        output->wavesCount = 1.25F;
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.capability_get = [](const cublasLtMatmulAlgo_t* algorithm,
                                  cublasLtMatmulAlgoCapAttributes_t attribute, void* output,
                                  std::size_t bytes, std::size_t* written) {
        expect(algorithm->data[0] == 77 && attribute == CUBLASLT_ALGO_CAP_POINTER_MODE_MASK,
               "capability SDK receives only the recorded original algorithm and attribute");
        if (!bytes && written) {
            *written = 4;
            return CUBLAS_STATUS_SUCCESS;
        }
        if (!output || bytes != 4) return CUBLAS_STATUS_INVALID_VALUE;
        const std::uint32_t mask = 3;
        std::memcpy(output, &mask, sizeof(mask));
        if (written) *written = sizeof(mask);
        return CUBLAS_STATUS_SUCCESS;
    };
    result.lt.matmul = [](cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
                          const void* alpha, const void* a, cublasLtMatrixLayout_t, const void* b,
                          cublasLtMatrixLayout_t, const void* beta, const void* c,
                          cublasLtMatrixLayout_t, void* d, cublasLtMatrixLayout_t,
                          const cublasLtMatmulAlgo_t* algorithm, void* workspace, std::size_t bytes,
                          cudaStream_t stream) {
        expect(chain_current == chain.handle_context &&
                   handle == reinterpret_cast<cublasLtHandle_t>(&chain.handle) && a && b && c &&
                   d && c != d && algorithm && algorithm->data[0] == 77 && workspace &&
                   bytes == 64 && !stream,
               "native matmul retains independent C/D, workspace and exact algorithm");
        auto* object = reinterpret_cast<Object*>(operation);
        CUdeviceptr bias{};
        std::memcpy(&bias, object->attributes[CUBLASLT_MATMUL_DESC_BIAS_POINTER].data(),
                    sizeof(bias));
        const auto* av = static_cast<const float*>(a);
        const auto* bv = static_cast<const float*>(b);
        const auto* cv = static_cast<const float*>(c);
        auto* dv = static_cast<float*>(d);
        const auto* bias_values = reinterpret_cast<const float*>(bias);
        expect(
            *static_cast<const float*>(alpha) == 2.0F && *static_cast<const float*>(beta) == 0.5F,
            "raw copied/device scalar values preserve their original SDK meaning");
        for (std::size_t column = 0; column < 2; ++column)
            for (std::size_t row = 0; row < 2; ++row)
                dv[row + column * 4] =
                    2.0F * (av[row] * bv[column * 2] + av[row + 2] * bv[column * 2 + 1]) +
                    0.5F * cv[row + column * 3] + bias_values[row];
        expect(dv[0] == 15.5F && dv[1] == 23.0F && dv[4] == 33.0F && dv[5] == 48.5F && dv[2] == 3 &&
                   dv[3] == 4 && dv[6] == 7 && dv[7] == 8 && cv[0] == 1 && cv[3] == 4,
               "injected SDK observes independent C/D strides, bias, original C and D padding");
        return ++chain.matmuls == 1 ? CUBLAS_STATUS_EXECUTION_FAILED : CUBLAS_STATUS_SUCCESS;
    };
    result.random.create = [](curandGenerator_t* output, curandRngType_t type) {
        expect(chain_current && type == CURAND_RNG_PSEUDO_PHILOX4_32_10,
               "original random factory uses the actual server context and RNG type");
        if (++random_native.creates == 1) return CURAND_STATUS_ALLOCATION_FAILED;
        const auto found = std::ranges::find(random_native.live, false);
        expect(found != random_native.live.end(), "bounded random provider slots");
        const auto index = static_cast<std::size_t>(found - random_native.live.begin());
        random_native.live[index] = true;
        *output = reinterpret_cast<curandGenerator_t>(&random_native.handles[index]);
        return CURAND_STATUS_SUCCESS;
    };
    result.random.set_seed = [](curandGenerator_t handle, unsigned long long seed) {
        expect(chain_current &&
                   handle == reinterpret_cast<curandGenerator_t>(&random_native.handles[0]),
               "seed uses actual native handle, not frontend opaque token");
        ++random_native.seeds;
        random_native.seed = seed;
        return seed == 13 ? CURAND_STATUS_TYPE_ERROR : CURAND_STATUS_SUCCESS;
    };
    result.random.destroy = [](curandGenerator_t handle) {
        expect(chain_current, "random destruction stays on native context owner");
        std::size_t index{};
        while (index < random_native.handles.size() &&
               handle != reinterpret_cast<curandGenerator_t>(&random_native.handles[index]))
            ++index;
        expect(index < random_native.handles.size() && random_native.live[index],
               "destroy one known live generator exactly once");
        random_native.live[index] = false;
        ++random_native.destroys;
        return random_native.quarantine ? CURAND_STATUS_INTERNAL_ERROR : CURAND_STATUS_SUCCESS;
    };
    return result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
void application_random(const char* provider, bool quarantine) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent original random provider");
    const auto create = reinterpret_cast<decltype(&curandCreateGenerator)>(
        ::dlsym(library, "curandCreateGenerator"));
    const auto seed = reinterpret_cast<decltype(&curandSetPseudoRandomGeneratorSeed)>(
        ::dlsym(library, "curandSetPseudoRandomGeneratorSeed"));
    const auto destroy = reinterpret_cast<decltype(&curandDestroyGenerator)>(
        ::dlsym(library, "curandDestroyGenerator"));
    expect(create == &curandCreateGenerator && seed == &curandSetPseudoRandomGeneratorSeed &&
               destroy == &curandDestroyGenerator &&
               ::dlsym(RTLD_DEFAULT, "curandCreateGenerator") == reinterpret_cast<void*>(create) &&
               ::dlsym(library, "curandGenerateUniform") ==
                   reinterpret_cast<void*>(&curandGenerateUniform),
           "direct/global/explicit random symbols route to controlled ABI");
    (void)::dlerror();
    expect(!::dlsym(library, "curandUnknown") && ::dlerror(),
           "unknown explicit random entry produces a real loader error");
    CUcontext context{};
    expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS, "create owned random context");
    auto handle = reinterpret_cast<curandGenerator_t>(std::uintptr_t{17});
    expect(create(&handle, CURAND_RNG_PSEUDO_PHILOX4_32_10) == CURAND_STATUS_ALLOCATION_FAILED &&
               handle == reinterpret_cast<curandGenerator_t>(std::uintptr_t{17}),
           "original known-empty failure preserves caller output and exact cuRAND status");
    expect(create(nullptr, CURAND_RNG_PSEUDO_PHILOX4_32_10) == CURAND_STATUS_OUT_OF_RANGE &&
               create(&handle, CURAND_RNG_PSEUDO_PHILOX4_32_10) == CURAND_STATUS_SUCCESS,
           "invalid caller storage rejects; fresh generator creation succeeds");
    expect(seed(handle, 13) == CURAND_STATUS_TYPE_ERROR &&
               seed(handle, UINT64_MAX - 17) == CURAND_STATUS_SUCCESS,
           "SDK seed error and exact full-width successful seed cross actual IPC");
    expect(curandGenerateUniform(handle, nullptr, 0) == CURAND_STATUS_TYPE_ERROR &&
               curandGenerateSeeds(handle) == CURAND_STATUS_TYPE_ERROR &&
               curandSetGeneratorOffset(handle, 0) == CURAND_STATUS_TYPE_ERROR,
           "unsupported generator operations cannot fall through to client-native SDK");
    expect(cuCtxSetCurrent(nullptr) == CUDA_SUCCESS &&
               seed(handle, 0) == CURAND_STATUS_NOT_INITIALIZED &&
               cuCtxSetCurrent(context) == CUDA_SUCCESS,
           "generator cannot leave its owning current context");
    const auto child = ::fork();
    expect(child >= 0, "random frontend fork guard");
    if (!child) ::_exit(seed(handle, 0) == CURAND_STATUS_NOT_INITIALIZED ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "forked frontend cannot send inherited generator commands");
    expect(::dlclose(library) == 0 && seed(handle, UINT64_MAX - 17) == CURAND_STATUS_SUCCESS,
           "cached wrapper survives provider close without retaining native code pointers");
    if (quarantine) {
        expect(destroy(handle) == CURAND_STATUS_INTERNAL_ERROR &&
                   destroy(handle) == CURAND_STATUS_NOT_INITIALIZED &&
                   seed(handle, 0) == CURAND_STATUS_NOT_INITIALIZED,
               "ambiguous destruction cannot be retried or reused");
        return;
    }
    expect(destroy(handle) == CURAND_STATUS_SUCCESS &&
               destroy(handle) == CURAND_STATUS_NOT_INITIALIZED &&
               seed(handle, 0) == CURAND_STATUS_NOT_INITIALIZED,
           "retired generator token cannot alias a later native slot");
    curandGenerator_t remaining{};
    expect(curandCreateGenerator(&remaining, CURAND_RNG_PSEUDO_PHILOX4_32_10) ==
                   CURAND_STATUS_SUCCESS &&
               remaining != handle && cuCtxDestroy(context) == CUDA_SUCCESS &&
               seed(remaining, 0) == CURAND_STATUS_NOT_INITIALIZED,
           "context cleanup destroys the outstanding generator and invalidates its token");
}
void application_chain(const char* provider, bool uses_blas_alias) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load original Lt chain provider");
    expect(::dlsym(library, "cublasLtMatmul") == reinterpret_cast<void*>(&cublasLtMatmul) &&
               ::dlsym(RTLD_DEFAULT, "cublasLtMatmulAlgoGetHeuristic") ==
                   reinterpret_cast<void*>(&cublasLtMatmulAlgoGetHeuristic),
           "chain uses direct/global/explicit public ABI");
    const auto driver = [](CUresult result,
                           std::source_location location = std::source_location::current()) {
        if (result != CUDA_SUCCESS)
            throw std::runtime_error(
                "owned chain Driver operation: status=" + std::to_string(result) +
                "; line=" + std::to_string(location.line()));
    };
    CUcontext context{};
    driver(cuCtxCreate(&context, 0, 0));
    cublasLtHandle_t handle{};
    cublasHandle_t blas_handle{};
    if (uses_blas_alias) {
        check(cublasCreate_v2(&blas_handle));
        handle = reinterpret_cast<cublasLtHandle_t>(blas_handle);
    } else
        check(cublasLtCreate(&handle));
    cublasLtMatrixLayout_t ab{}, c{}, d{};
    expect(cublasLtMatrixLayoutCreate(&ab, CUDA_R_32F, 2, 2, 2) == CUBLAS_STATUS_ALLOC_FAILED,
           "original descriptor failure is preserved before retry");
    check(cublasLtMatrixLayoutCreate(&ab, CUDA_R_32F, 2, 2, 2));
    check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, 2, 2, 3));
    check(cublasLtMatrixLayoutCreate(&d, CUDA_R_32F, 2, 2, 4));
    cublasLtMatmulDesc_t operation{};
    cublasLtMatmulPreference_t preference{};
    check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    check(cublasLtMatmulPreferenceCreate(&preference));
    std::array<CUdeviceptr, 6> allocations{};
    for (auto& pointer : allocations) driver(cuMemAlloc(&pointer, 64));
    const auto* bias = reinterpret_cast<const void*>(allocations[4]);
    const auto epilogue = CUBLASLT_EPILOGUE_BIAS;
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                         sizeof(bias)));
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                         sizeof(epilogue)));
    std::array<cublasLtMatmulHeuristicResult_t, 4> algorithms{};
    algorithms[0].workspaceSize = 123;
    int count = 19;
    expect(
        cublasLtMatmulAlgoGetHeuristic(handle, operation, ab, ab, c, d, preference, 4,
                                       algorithms.data(), &count) == CUBLAS_STATUS_NOT_SUPPORTED &&
            count == 19 && algorithms[0].workspaceSize == 123,
        "original heuristic failure preserves output bytes and count");
    check(cublasLtMatmulAlgoGetHeuristic(handle, operation, ab, ab, c, d, preference, 4,
                                         algorithms.data(), &count));
    expect(count == 1 && algorithms[0].algo.data[0] == 77 && algorithms[0].workspaceSize == 64 &&
               algorithms[0].wavesCount == 2.5F && algorithms[0].reserved[0] == -7 &&
               !algorithms[1].workspaceSize,
           "explicit codec preserves original algorithm, fields and unwritten result tail");
    const auto capability = reinterpret_cast<decltype(&cublasLtMatmulAlgoCapGetAttribute)>(
        ::dlsym(library, "cublasLtMatmulAlgoCapGetAttribute"));
    expect(capability == &cublasLtMatmulAlgoCapGetAttribute &&
               ::dlsym(RTLD_DEFAULT, "cublasLtMatmulAlgoCapGetAttribute") ==
                   reinterpret_cast<void*>(capability),
           "all public capability routes stay owned");
    for (const auto get : {&cublasLtMatmulAlgoCapGetAttribute, capability}) {
        std::uint32_t mask = 0xa5a5a5a5;
        std::size_t written = 99;
        check(get(&algorithms[0].algo, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, &mask, sizeof(mask),
                  &written));
        expect(mask == 3 && written == 4, "actual owner capability, not caller-native fallback");
        check(get(&algorithms[0].algo, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, nullptr, 0, &written));
        expect(written == 4, "zero-size capability query preserves native count");
        auto forged = algorithms[0].algo;
        forged.data[0] = 78;
        mask = 19;
        expect(get(&forged, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, &mask, 4, &written) ==
                       CUBLAS_STATUS_NOT_SUPPORTED &&
                   mask == 19,
               "unknown algorithm cannot enter the original metadata SDK");
    }
    cublasLtMatmulHeuristicResult_t checked{};
    checked.algo.data[0] = 999;
    check(cublasLtMatmulAlgoCheck(handle, operation, ab, ab, c, d, &algorithms[0].algo, &checked));
    expect(
        checked.algo.data[0] == 999 && checked.workspaceSize == 64 && checked.wavesCount == 1.25F,
        "AlgoCheck must not update a distinct output algorithm field");
    auto forged = algorithms[0].algo;
    forged.data[0] = 78;
    expect(cublasLtMatmulAlgoCheck(handle, operation, ab, ab, c, d, &forged, &checked) ==
               CUBLAS_STATUS_NOT_SUPPORTED,
           "unknown algorithm bytes never reach an original SDK executable entry");
    const float alpha = 2, beta = 0.5F;
    expect(cublasLtMatmul(handle, operation, &alpha, reinterpret_cast<const void*>(allocations[0]),
                          ab, reinterpret_cast<const void*>(allocations[1]), ab, &beta,
                          reinterpret_cast<const void*>(allocations[2]), c,
                          reinterpret_cast<void*>(allocations[3]), d, &algorithms[0].algo,
                          reinterpret_cast<void*>(allocations[5]), 64,
                          nullptr) == CUBLAS_STATUS_EXECUTION_FAILED,
           "actual original matmul error is not an admission acknowledgment");
    auto destroy =
        std::async(std::launch::async, [&] { return cublasLtMatmulDescDestroy(operation); });
    const auto pending =
        destroy.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    chain.can_stop.store(true);
    const auto destroyed = destroy.get();
    expect(pending && destroyed == CUBLAS_STATUS_SUCCESS,
           "immutable descriptor remains retained until independent physical stop");
    driver(cuCtxSynchronize());
    check(cublasLtMatrixLayoutDestroy(ab));
    check(cublasLtMatrixLayoutDestroy(c));
    check(cublasLtMatrixLayoutDestroy(d));
    check(cublasLtMatmulPreferenceDestroy(preference));
    if (uses_blas_alias) {
        expect(cublasLtDestroy(handle) == CUBLAS_STATUS_NOT_INITIALIZED,
               "compute alias cannot acquire a second native destruction owner");
        check(cublasDestroy_v2(blas_handle));
    } else
        check(cublasLtDestroy(handle));
    for (auto pointer : allocations) driver(cuMemFree(pointer));
    driver(cuCtxDestroy(context));
    std::uint32_t retained_mask{};
    check(capability(&algorithms[0].algo, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, &retained_mask,
                     sizeof(retained_mask), nullptr));
    expect(retained_mask == 3,
           "algorithm value query survives native handle and GPU context retirement");
    expect(glimmer_lt_native_calls() == 0 && ::dlclose(library) == 0,
           "public Lt chain never invokes the caller-native provider");
}
template <typename Function>
Function symbol(void* library, const char* name, Function direct) {
    const auto value = ::dlsym(library, name);
    expect(
        value && value == ::dlsym(RTLD_DEFAULT, name) && value == reinterpret_cast<void*>(direct),
        "direct/global/explicit Lt symbol identity");
    return reinterpret_cast<Function>(value);
}
void application(const char* provider, bool quarantine) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library != nullptr, "load independent original Lt provider");
#define SELECT(NAME) const auto NAME##_call = symbol(library, #NAME, &(NAME))
    SELECT(cublasLtMatrixLayoutCreate);
    SELECT(cublasLtMatrixLayoutDestroy);
    SELECT(cublasLtMatrixLayoutSetAttribute);
    SELECT(cublasLtMatrixLayoutGetAttribute);
    SELECT(cublasLtMatmulDescCreate);
    SELECT(cublasLtMatmulDescDestroy);
    SELECT(cublasLtMatmulDescSetAttribute);
    SELECT(cublasLtMatmulDescGetAttribute);
    SELECT(cublasLtMatmulPreferenceCreate);
    SELECT(cublasLtMatmulPreferenceDestroy);
    SELECT(cublasLtMatmulPreferenceSetAttribute);
    SELECT(cublasLtMatmulPreferenceGetAttribute);
    SELECT(cublasLtGetVersion);
    SELECT(cublasLtGetCudartVersion);
    SELECT(cublasLtCreate);
    SELECT(cublasLtDestroy);
    SELECT(cublasLtMatmulAlgoGetHeuristic);
    SELECT(cublasLtMatmulAlgoCheck);
    SELECT(cublasLtMatrixTransform);
    SELECT(cublasLtMatmul);
    SELECT(cublasLtMatrixLayoutInit_internal);
    SELECT(cublasLtMatmulDescInit_internal);
    SELECT(cublasLtMatmulPreferenceInit_internal);
#undef SELECT
    expect(
        cublasLtCreate_call(nullptr) == CUBLAS_STATUS_INVALID_VALUE &&
            cublasLtDestroy_call(nullptr) == CUBLAS_STATUS_NOT_INITIALIZED &&
            cublasLtMatmulAlgoGetHeuristic_call(nullptr, nullptr, nullptr, nullptr, nullptr,
                                                nullptr, nullptr, 0, nullptr,
                                                nullptr) == CUBLAS_STATUS_NOT_INITIALIZED &&
            cublasLtMatmulAlgoCheck_call(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                         nullptr, nullptr) == CUBLAS_STATUS_INVALID_VALUE &&
            cublasLtMatrixTransform_call(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                         nullptr, nullptr, nullptr, nullptr,
                                         nullptr) == CUBLAS_STATUS_NOT_SUPPORTED &&
            cublasLtMatmul_call(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                                nullptr) == CUBLAS_STATUS_NOT_INITIALIZED &&
            cublasLtMatrixLayoutInit_internal_call(nullptr, 0, CUDA_R_32F, 0, 0, 0) ==
                CUBLAS_STATUS_NOT_SUPPORTED &&
            cublasLtMatmulDescInit_internal_call(nullptr, 0, CUBLAS_COMPUTE_32F, CUDA_R_32F) ==
                CUBLAS_STATUS_NOT_SUPPORTED &&
            cublasLtMatmulPreferenceInit_internal_call(nullptr, 0) == CUBLAS_STATUS_NOT_SUPPORTED,
        "unattached GPU/stack-descriptor APIs cannot escape to caller SDK");
    CUcontext current{};
    expect(cuCtxGetCurrent(&current) == CUDA_ERROR_NOT_INITIALIZED && !current,
           "no initial application CUDA context");
    const auto actual_version = cublasLtGetVersion_call();
    const auto actual_runtime = cublasLtGetCudartVersion_call();
    if (actual_version != 120604 || actual_runtime != 12031)
        throw std::runtime_error("actual Lt/runtime versions: " + std::to_string(actual_version) +
                                 "/" + std::to_string(actual_runtime));
    cublasLtMatrixLayout_t layout{};
    expect(cublasLtMatrixLayoutCreate_call(nullptr, CUDA_R_32F, 3, 5, 4) ==
               CUBLAS_STATUS_INVALID_VALUE,
           "bad output rejected before SDK");
    expect(cublasLtMatrixLayoutCreate_call(&layout, CUDA_R_32F, 3, 5, 4) ==
                   CUBLAS_STATUS_ALLOC_FAILED &&
               !layout,
           "known-empty SDK create preserves status/output and permits fresh retry");
    check(cublasLtMatrixLayoutCreate_call(&layout, CUDA_R_32F, 3, 5, 4));
    std::size_t written{91};
    if (quarantine) {
        std::uint64_t rows{99};
        expect(cublasLtMatrixLayoutGetAttribute_call(layout, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows,
                                                     sizeof(rows),
                                                     &written) == CUBLAS_STATUS_INTERNAL_ERROR &&
                   rows == 99 && written == 91,
               "inconsistent successful SDK query cannot publish partial output");
        expect(cublasLtGetVersion_call() == 0 &&
                   cublasLtMatrixLayoutDestroy_call(layout) == CUBLAS_STATUS_NOT_INITIALIZED,
               "quarantined metadata cannot replay or retry cleanup");
        expect(::dlclose(library) == 0, "release caller provider");
        return;
    }
    check(cublasLtMatrixLayoutGetAttribute_call(layout, CUBLASLT_MATRIX_LAYOUT_ROWS, nullptr, 0,
                                                &written));
    expect(written == 8, "actual SDK zero-buffer size query");
    std::uint32_t order{CUBLASLT_ORDER_ROW};
    check(cublasLtMatrixLayoutSetAttribute_call(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                                sizeof(order)));
    order = 77;
    expect(cublasLtMatrixLayoutSetAttribute_call(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                                 sizeof(order)) == CUBLAS_STATUS_INVALID_VALUE,
           "original invalid attribute status survives transport");
    const auto page = ::sysconf(_SC_PAGESIZE);
    expect(page > 0, "page extent");
    auto* output = ::mmap(nullptr, static_cast<std::size_t>(page), PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(output != MAP_FAILED, "write-only attribute output");
    check(cublasLtMatrixLayoutGetAttribute_call(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, output, 4,
                                                &written));
    expect(::mprotect(output, static_cast<std::size_t>(page), PROT_READ) == 0 && written == 4 &&
               *static_cast<const std::uint32_t*>(output) == CUBLASLT_ORDER_ROW &&
               ::munmap(output, static_cast<std::size_t>(page)) == 0,
           "query never reads previous output bytes");
    std::uint64_t rows{99};
    written = 91;
    expect(cublasLtMatrixLayoutGetAttribute_call(layout, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, 4,
                                                 &written) == CUBLAS_STATUS_INVALID_VALUE &&
               rows == 99 && written == 91,
           "native query failure preserves both outputs");
    cublasLtMatmulDesc_t operation{};
    check(cublasLtMatmulDescCreate_call(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    std::int8_t fast{1};
    expect(cublasLtMatmulDescSetAttribute_call(operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast,
                                               sizeof(fast)) == CUBLAS_STATUS_SUCCESS,
           "one-byte FAST_ACCUM setter must reach the original SDK");
    fast = 0;
    check(cublasLtMatmulDescGetAttribute_call(operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast,
                                              sizeof(fast), &written));
    expect(fast == 1 && written == 1, "actual SDK byte-sized fast-accumulation flag");
    std::uint32_t mode{CUBLASLT_POINTER_MODE_DEVICE};
    check(cublasLtMatmulDescSetAttribute_call(operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode,
                                              4));
    mode = 0;
    check(cublasLtMatmulDescGetAttribute_call(operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode,
                                              4, &written));
    expect(mode == CUBLASLT_POINTER_MODE_DEVICE && written == 4, "actual operation scalar mode");
    const void* bias{};
    check(cublasLtMatmulDescSetAttribute_call(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                              sizeof(bias)));
    std::uint64_t unowned{0x12345678};
    expect(
        cublasLtMatmulDescSetAttribute_call(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &unowned,
                                            sizeof(unowned)) == CUBLAS_STATUS_INVALID_VALUE,
        "unowned device pointer cannot become service authority");
    expect(cublasLtMatmulDescSetAttribute_call(operation, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                               &unowned,
                                               sizeof(unowned)) == CUBLAS_STATUS_NOT_SUPPORTED,
           "unclassified pointer attributes never reach native SDK");
    cublasLtMatmulPreference_t preference{};
    check(cublasLtMatmulPreferenceCreate_call(&preference));
    cublasLtMatrixLayout_t excess{};
    expect(cublasLtMatrixLayoutCreate_call(&excess, CUDA_R_32F, 3, 5, 4) ==
                   CUBLAS_STATUS_ALLOC_FAILED &&
               !excess && cublasLtGetVersion_call() == 120604,
           "known pre-SDK object limit does not quarantine existing resources");
    std::uint64_t workspace{4096};
    check(cublasLtMatmulPreferenceSetAttribute_call(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace)));
    workspace = 0;
    check(cublasLtMatmulPreferenceGetAttribute_call(preference,
                                                    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspace, sizeof(workspace), &written));
    expect(workspace == 4096 && written == 8,
           "native preference retains explicit workspace budget");
    check(cublasLtMatmulPreferenceDestroy_call(preference));
    check(cublasLtMatmulDescDestroy_call(operation));
    const auto stale = layout;
    check(cublasLtMatrixLayoutDestroy_call(layout));
    expect(cublasLtMatrixLayoutDestroy_call(stale) == CUBLAS_STATUS_NOT_INITIALIZED,
           "destroyed opaque descriptor is not recycled");
    for (unsigned int iteration = 0; iteration < 17000; ++iteration) {
        check(cublasLtMatrixLayoutCreate_call(&layout, CUDA_R_32F, 3, 5, 4));
        expect(layout != stale, "native address reuse never aliases retired public token");
        check(cublasLtMatrixLayoutDestroy_call(layout));
    }
    expect(cuCtxGetCurrent(&current) == CUDA_ERROR_NOT_INITIALIZED && !current &&
               glimmer_lt_native_calls() == 0,
           "CPU metadata neither creates app context nor calls caller SDK");
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "same-process role control identity");
    glimmer::control::ExecutionSocketClient role_control(
        {std::getenv("GLIMMER_EXECUTION_SOCKET"), *identity, 3000});
    glimmer::control::ExecutionPayloadWriter bootstrap;
    expect(
        bootstrap.write_u32(glimmer::control::kExecutionLtMetadataRole) && bootstrap.write_u32(0),
        "explicit CPU role");
    const auto opened = role_control.open(bootstrap.take());
    expect(opened && opened->status == 0 &&
               glimmer::control::decode_execution_bootstrap_ownership(opened->data) ==
                   glimmer::control::ExecutionBootstrapOwnership::kMetadataRetained,
           "CPU ownership is distinct from GPU context proof");
    for (const auto operation :
         {glimmer::control::ExecutionOpcode::kAllocate, glimmer::control::ExecutionOpcode::kLaunch,
          glimmer::control::ExecutionOpcode::kCreateBlas,
          glimmer::control::ExecutionOpcode::kCreateLt,
          glimmer::control::ExecutionOpcode::kDestroyLt}) {
        const auto rejected = role_control.exchange(operation);
        expect(
            rejected && rejected->status == CUDA_ERROR_NOT_SUPPORTED && !rejected->original_status,
            "metadata namespace cannot issue GPU work or acquire library handles");
    }
    const auto closed = role_control.exchange(glimmer::control::ExecutionOpcode::kClose);
    expect(closed && closed->status == 0, "close independent CPU-only owner");
    expect(::dlsym(library, "cublasLtUnknownOwnedEntry") == nullptr && ::dlerror(),
           "unknown explicit SDK symbol fails closed");
    expect(::dlclose(library) == 0, "release caller SDK provider");
}
void application_runtime_metadata(const char* driver_path, const char* runtime_path) {
    auto* driver = ::dlopen(driver_path, RTLD_NOW | RTLD_LOCAL);
    auto* runtime = ::dlopen(runtime_path, RTLD_NOW | RTLD_LOCAL);
    expect(driver && runtime, "load independent caller SDK metadata providers");
    expect(::dlsym(driver, "cuDeviceGetCount") == reinterpret_cast<void*>(&cuDeviceGetCount),
           "anchor original Driver metadata without a native caller context");
    expect(::dlsym(driver, "cuModuleGetGlobal_v2") == reinterpret_cast<void*>(&cuModuleGetGlobal),
           "explicit original Driver global lookup returns the owned modern ABI");
    const auto count = symbol(runtime, "cudaGetDeviceCount", &cudaGetDeviceCount);
    (void)symbol(runtime, "cudaGetSymbolAddress", &cudaGetSymbolAddress);
    (void)symbol(runtime, "cudaGetSymbolSize", &cudaGetSymbolSize);
    (void)symbol(runtime, "cudaMemcpyToSymbol", &cudaMemcpyToSymbol);
    (void)symbol(runtime, "cudaMemcpyFromSymbol", &cudaMemcpyFromSymbol);
    (void)symbol(runtime, "cudaMemcpyToSymbolAsync", &cudaMemcpyToSymbolAsync);
    (void)symbol(runtime, "cudaMemcpyFromSymbolAsync", &cudaMemcpyFromSymbolAsync);
    const auto driver_version = symbol(runtime, "cudaDriverGetVersion", &cudaDriverGetVersion);
    const auto properties = symbol(runtime, "cudaGetDeviceProperties_v2", &cudaGetDeviceProperties);
    const auto attribute = symbol(runtime, "cudaDeviceGetAttribute", &cudaDeviceGetAttribute);
    const auto peer = symbol(runtime, "cudaDeviceGetP2PAttribute", &cudaDeviceGetP2PAttribute);
    const auto peer_access = symbol(runtime, "cudaDeviceCanAccessPeer", &cudaDeviceCanAccessPeer);
    const auto priority_range =
        symbol(runtime, "cudaDeviceGetStreamPriorityRange", &cudaDeviceGetStreamPriorityRange);
    const auto memory = symbol(runtime, "cudaMemGetInfo", &cudaMemGetInfo);
    const auto create_stream =
        symbol(runtime, "cudaStreamCreateWithPriority", &cudaStreamCreateWithPriority);
    const auto stream_priority = symbol(runtime, "cudaStreamGetPriority", &cudaStreamGetPriority);
    const auto stream_flags = symbol(runtime, "cudaStreamGetFlags", &cudaStreamGetFlags);
    const auto capture = symbol(runtime, "cudaStreamIsCapturing", &cudaStreamIsCapturing);
    const auto capture_info =
        symbol(runtime, "cudaStreamGetCaptureInfo_v2", &cudaStreamGetCaptureInfo);
    const auto exchange_mode = symbol(runtime, "cudaThreadExchangeStreamCaptureMode",
                                      &cudaThreadExchangeStreamCaptureMode);
    const auto native_count = reinterpret_cast<unsigned int (*)(unsigned int)>(
        ::dlsym(driver, "glimmer_execution_lookup_count"));
    const auto runtime_count = reinterpret_cast<unsigned int (*)(unsigned int)>(
        ::dlsym(runtime, "glimmer_execution_runtime_lookup_count"));
    expect(native_count && runtime_count, "independent caller SDK operation counters");
    int devices{};
    int version{};
    expect(driver_version(&version) == cudaSuccess && version == 12080,
           "Runtime Driver version uses original Driver metadata without Runtime initialization");
    expect(count(&devices) == cudaSuccess && devices == 2 && native_count(2) == 0,
           "device count preserves the real Driver value without caller GPU resources");
    cudaDeviceProp queried{};
    expect(properties(&queried, 0) == cudaSuccess && queried.multiProcessorCount == 80 &&
               queried.major == 7 && std::string_view(queried.name) == "Owned device metadata",
           "complete SDK property values traverse owned worker and exact versioned ABI");
    queried.multiProcessorCount = 91;
    expect(properties(&queried, -1) == cudaErrorInvalidDevice &&
               queried.multiProcessorCount == 91 && cudaGetLastError() == cudaErrorInvalidDevice,
           "original Runtime error and unchanged caller output survive native partial writes");
    int value = 37;
    expect(attribute(&value, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess && value == 80,
           "original Runtime attribute value");
    expect(attribute(&value, cudaDevAttrWarpSize, 0) == cudaErrorCallRequiresNewerDriver &&
               value == 80 && cudaGetLastError() == cudaErrorCallRequiresNewerDriver,
           "original Runtime status 36 is not converted through the Driver error domain");
    expect(peer(&value, cudaDevP2PAttrAccessSupported, 0, 1) == cudaSuccess && value == 1,
           "owned original peer-attribute query");
    const auto before = chain.metadata_calls.load();
    chain.runtime_version.store(12050);
    expect(properties(&queried, 0) == cudaErrorNotSupported && queried.multiProcessorCount == 91 &&
               chain.metadata_calls.load() == before,
           "incompatible SDK ABI rejects before the original property target or output write");
    chain.runtime_version.store(12060);
    const auto extent = ::sysconf(_SC_PAGESIZE);
    expect(extent > 0 && static_cast<std::size_t>(extent) >= sizeof(cudaDeviceProp),
           "property output page size");
    auto* output = ::mmap(nullptr, static_cast<std::size_t>(extent), PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(
        output != MAP_FAILED && properties(static_cast<cudaDeviceProp*>(output), 0) == cudaSuccess,
        "write-only metadata output does not require reading its old bytes");
    expect(::mprotect(output, static_cast<std::size_t>(extent), PROT_READ) == 0 &&
               static_cast<cudaDeviceProp*>(output)->multiProcessorCount == 80 &&
               ::munmap(output, static_cast<std::size_t>(extent)) == 0,
           "complete write-only output and mapping cleanup");
    unsigned int flags{};
    int active{};
    expect(cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS && active == 0 &&
               native_count(2) == 0 && runtime_count(2) == 0,
           "metadata does not mint an application primary context or use caller-native Runtime GPU "
           "APIs");
    int least = 7, greatest = 7;
    expect(priority_range(&least, &greatest) == cudaSuccess && least == 0 && greatest == -2 &&
               priority_range(nullptr, nullptr) == cudaSuccess,
           "original stream priority limits and optional outputs");
    expect(peer_access(&value, 0, 0) == cudaSuccess && value == 0,
           "same-device peer access preserves original metadata");
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    expect(exchange_mode(&mode) == cudaSuccess && mode == cudaStreamCaptureModeGlobal,
           "capture-mode exchange returns previous TLS state");
    auto other_mode = std::async(std::launch::async, [&] {
        auto other = cudaStreamCaptureModeThreadLocal;
        expect(exchange_mode(&other) == cudaSuccess, "independent thread capture mode");
        return other;
    });
    expect(other_mode.get() == cudaStreamCaptureModeGlobal, "capture-mode state is thread-local");
    mode = static_cast<cudaStreamCaptureMode>(3);
    expect(exchange_mode(&mode) == cudaErrorInvalidValue && static_cast<int>(mode) == 3 &&
               cudaGetLastError() == cudaErrorInvalidValue,
           "invalid capture mode does not change state or caller output");
    mode = cudaStreamCaptureModeGlobal;
    expect(exchange_mode(&mode) == cudaSuccess && mode == cudaStreamCaptureModeRelaxed,
           "rejected exchange retained previous TLS state");
    cudaStream_t stream{};
    expect(create_stream(&stream, cudaStreamNonBlocking, -17) == cudaSuccess && stream,
           "owned original priority stream creation");
    int priority = 73;
    expect(stream_priority(stream, &priority) == cudaSuccess && priority == -2 &&
               stream_flags(stream, &flags) == cudaSuccess && flags == cudaStreamNonBlocking,
           "SDK-clamped priority and immutable flags, not unchecked requested priority");
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusActive;
    expect(capture(stream, &capturing) == cudaSuccess && capturing == cudaStreamCaptureStatusNone,
           "owned noncapturing stream observation");
    unsigned long long capture_id = 99;
    cudaGraph_t graph = reinterpret_cast<cudaGraph_t>(17);
    const cudaGraphNode_t* dependencies = reinterpret_cast<const cudaGraphNode_t*>(23);
    std::size_t dependency_count = 77;
    expect(
        capture_info(stream, &capturing, &capture_id, &graph, &dependencies, &dependency_count) ==
                cudaSuccess &&
            capturing == cudaStreamCaptureStatusNone && capture_id == 99 &&
            graph == reinterpret_cast<cudaGraph_t>(17) &&
            dependencies == reinterpret_cast<const cudaGraphNode_t*>(23) && dependency_count == 77,
        "noncapturing capture-info publishes only its defined status output");
    expect(capture(cudaStreamLegacy, &capturing) == cudaSuccess &&
               stream_flags(cudaStreamLegacy, &flags) == cudaSuccess && flags == 0,
           "explicit legacy identity normalizes to the real legacy default stream");
    expect(capture(cudaStreamPerThread, &capturing) == cudaSuccess &&
               stream_flags(cudaStreamPerThread, &flags) == cudaSuccess && flags == 0 &&
               cudaStreamQuery(cudaStreamPerThread) == cudaSuccess && chain.stream_creates == 2,
           "one caller-thread proxy is reused across observation and completion");
    auto thread_proxy = std::async(std::launch::async, [&] {
        cudaStreamCaptureStatus status{};
        expect(capture(cudaStreamPerThread, &status) == cudaSuccess &&
                   cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess,
               "second caller thread owns a distinct default-stream proxy");
    });
    thread_proxy.get();
    expect(chain.stream_creates == 3 && chain.stream_releases == 0,
           "thread exit does not destroy a potentially live native proxy");
    std::size_t available{}, total{};
    expect(memory(&available, &total) == cudaSuccess && available == 1234567 && total == 7654321,
           "current application-session memory capacity, not fabricated quota headroom");
    expect(cudaStreamDestroy(stream) == cudaSuccess, "original stream is destroyed once");
    priority = 73;
    expect(stream_priority(stream, &priority) == cudaErrorInvalidResourceHandle && priority == 73 &&
               cudaGetLastError() == cudaErrorInvalidResourceHandle,
           "retired stream cannot expose cached stale metadata");
    expect(
        cudaDeviceReset() == cudaSuccess && chain.stream_creates == 3 && chain.stream_releases == 3,
        "owned priority stream and Runtime context retire exactly once");
    expect(capture(cudaStreamPerThread, &capturing) == cudaSuccess && chain.stream_creates == 4 &&
               cudaDeviceReset() == cudaSuccess && chain.stream_releases == 4,
           "reset generation cannot reuse a retired per-thread default stream");
    expect(::dlclose(runtime) == 0 && ::dlclose(driver) == 0, "balance SDK metadata providers");
}
// These deprecated entry points are the compatibility behavior under test.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
void application_context_controls(const char* driver_path, const char* runtime_path) {
    auto* driver = ::dlopen(driver_path, RTLD_NOW | RTLD_LOCAL);
    auto* runtime = ::dlopen(runtime_path, RTLD_NOW | RTLD_LOCAL);
    expect(driver && runtime, "load independent context-control providers");
    const auto get_limit = symbol(driver, "cuCtxGetLimit", &cuCtxGetLimit);
    const auto runtime_get_limit = symbol(runtime, "cudaDeviceGetLimit", &cudaDeviceGetLimit);
    const auto native_count = reinterpret_cast<unsigned int (*)(unsigned int)>(
        ::dlsym(driver, "glimmer_execution_lookup_count"));
    const auto runtime_count = reinterpret_cast<unsigned int (*)(unsigned int)>(
        ::dlsym(runtime, "glimmer_execution_runtime_lookup_count"));
    expect(native_count && runtime_count &&
               ::dlsym(RTLD_DEFAULT, "cudaThreadGetLimit") ==
                   reinterpret_cast<void*>(&cudaThreadGetLimit),
           "global/explicit ABI routes cover original and legacy context controls");
    void* cached{};
    CUdriverProcAddressQueryResult found{};
    expect(cuGetProcAddress("cuCtxGetLimit", &cached, 12000, 0, &found) == CUDA_SUCCESS &&
               found == CU_GET_PROC_ADDRESS_SUCCESS && cached == reinterpret_cast<void*>(get_limit),
           "Driver cached lookup selects the same owned context wrapper");
    std::size_t limit = 55;
    expect(get_limit(&limit, CU_LIMIT_STACK_SIZE) == CUDA_ERROR_INVALID_CONTEXT && limit == 55,
           "Driver control does not implicitly create a context");
    CUcontext first{}, second{};
    expect(cuCtxCreate(&first, 0, 0) == CUDA_SUCCESS &&
               cuCtxSetLimit(CU_LIMIT_STACK_SIZE, UINT64_MAX - 17) == CUDA_SUCCESS &&
               runtime_get_limit(&limit, cudaLimitStackSize) == cudaSuccess &&
               limit == UINT64_MAX - 17 &&
               cudaDeviceSetLimit(cudaLimitStackSize, 4096) == cudaSuccess &&
               get_limit(&limit, CU_LIMIT_STACK_SIZE) == CUDA_SUCCESS && limit == 4096,
           "Driver and Runtime share the actual owned configuration without narrowing size_t");
    CUmodule module{};
    CUfunction function{};
    constexpr char image[] =
        ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry noop() { ret; }\n";
    expect(
        cuModuleLoadData(&module, image) == CUDA_SUCCESS &&
            cuModuleGetFunction(&function, module, "noop") == CUDA_SUCCESS &&
            cuLaunchKernel(function, 80, 1, 1, 32, 1, 1, 0, nullptr, nullptr, nullptr) ==
                CUDA_SUCCESS &&
            get_limit(&limit, CU_LIMIT_STACK_SIZE) == CUDA_SUCCESS && limit == 4096 &&
            !chain.can_stop.load(),
        "actual IPC query returns while the preceding injected command remains physically pending");
    chain.can_stop.store(true);
    expect(cuModuleUnload(module) == CUDA_SUCCESS, "observed stop permits module retirement");
    limit = 55;
    expect(get_limit(&limit, CU_LIMIT_PRINTF_FIFO_SIZE) == CUDA_ERROR_UNSUPPORTED_LIMIT &&
               limit == 55 &&
               runtime_get_limit(&limit, cudaLimitPrintfFifoSize) == cudaErrorUnsupportedLimit &&
               limit == 55 && cudaGetLastError() == cudaErrorUnsupportedLimit &&
               cudaPeekAtLastError() == cudaSuccess &&
               cuCtxGetLimit(nullptr, CU_LIMIT_STACK_SIZE) == CUDA_ERROR_INVALID_VALUE,
           "original errors preserve outputs and Runtime error-slot semantics");
    CUfunc_cache driver_cache{};
    cudaFuncCache runtime_cache{};
    CUsharedconfig driver_shared{};
    cudaSharedMemConfig runtime_shared{};
    expect(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1) == CUDA_SUCCESS &&
               cudaDeviceGetCacheConfig(&runtime_cache) == cudaSuccess &&
               runtime_cache == cudaFuncCachePreferL1 &&
               cudaDeviceSetCacheConfig(cudaFuncCachePreferShared) == cudaSuccess &&
               cuCtxGetCacheConfig(&driver_cache) == CUDA_SUCCESS &&
               driver_cache == CU_FUNC_CACHE_PREFER_SHARED &&
               cuCtxSetSharedMemConfig(CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE) == CUDA_SUCCESS &&
               cudaDeviceGetSharedMemConfig(&runtime_shared) == cudaSuccess &&
               runtime_shared == cudaSharedMemBankSizeFourByte &&
               cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) == cudaSuccess &&
               cuCtxGetSharedMemConfig(&driver_shared) == CUDA_SUCCESS &&
               driver_shared == CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE,
           "original cache/shared-bank setters and queries cross both APIs");
    expect(cudaThreadSetLimit(cudaLimitStackSize, 8192) == cudaSuccess &&
               cudaThreadGetLimit(&limit, cudaLimitStackSize) == cudaSuccess && limit == 8192 &&
               cudaThreadSetCacheConfig(cudaFuncCachePreferEqual) == cudaSuccess &&
               cudaThreadGetCacheConfig(&runtime_cache) == cudaSuccess &&
               runtime_cache == cudaFuncCachePreferEqual &&
               cuCtxResetPersistingL2Cache() == CUDA_SUCCESS &&
               cudaCtxResetPersistingL2Cache() == cudaSuccess,
           "legacy aliases and both native persisting-cache reset routes");
    CUexecAffinityParam affinity{};
    int least = 11, greatest = 12;
    expect(cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT) == CUDA_SUCCESS &&
               affinity.type == CU_EXEC_AFFINITY_TYPE_SM_COUNT &&
               affinity.param.smCount.val == 80 &&
               cuCtxGetStreamPriorityRange(&least, &greatest) == CUDA_SUCCESS && least == 0 &&
               greatest == -2 && cuCtxGetStreamPriorityRange(nullptr, nullptr) == CUDA_SUCCESS,
           "native affinity and signed priority outputs, including optional outputs");
    unsigned int version{};
    expect(
        cuCtxCreate(&second, 0, 0) == CUDA_SUCCESS &&
            get_limit(&limit, CU_LIMIT_STACK_SIZE) == CUDA_SUCCESS && limit == 1024 &&
            cuCtxGetApiVersion(first, &version) == CUDA_SUCCESS && version == 3021 &&
            cuCtxGetApiVersion(nullptr, &version) == CUDA_SUCCESS && version == 3022,
        "explicit queries address the named owner and do not replace the caller current context");
    const auto child = ::fork();
    expect(child >= 0, "fork context-control exclusion");
    if (child == 0)
        ::_exit(cuCtxGetLimit(&limit, CU_LIMIT_STACK_SIZE) == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int child_status{};
    expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "fork cannot inherit an owned execution connection");
    expect(cuCtxDestroy(second) == CUDA_SUCCESS && cuCtxSetCurrent(first) == CUDA_SUCCESS &&
               get_limit(&limit, CU_LIMIT_STACK_SIZE) == CUDA_SUCCESS && limit == 8192 &&
               native_count(2) == 0 && runtime_count(2) == 0,
           "context isolation and zero caller-native resource calls");
    expect(::dlclose(driver) == 0 && ::dlclose(runtime) == 0 &&
               get_limit(&limit, CU_LIMIT_STACK_SIZE) == CUDA_SUCCESS && limit == 8192 &&
               cuCtxDestroy(first) == CUDA_SUCCESS &&
               cuCtxGetApiVersion(first, &version) == CUDA_ERROR_INVALID_CONTEXT,
           "cached wrappers survive provider close while retired context tokens do not");
    expect(cudaDeviceGetLimit(&limit, cudaLimitStackSize) == cudaSuccess && limit == 1024 &&
               cudaDeviceReset() == cudaSuccess,
           "Runtime-only control lazily owns a fresh primary context and resets it");
}
#pragma GCC diagnostic pop
void run(const char* provider, bool quarantine, bool is_chain, bool uses_blas_alias,
         const char* metadata_driver = nullptr, const char* metadata_runtime = nullptr,
         bool is_random = false, bool is_context = false) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-lt-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = is_chain ? 16 : 3;
    config.exit_when_idle = true;
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "authenticate real metadata endpoint");
    native.is_quarantine = quarantine;
    random_native.quarantine = quarantine;
    auto functions = is_chain || metadata_driver || is_random ? chain_dispatch() : dispatch();
    if (is_context) {
        namespace fixture = glimmer::tests::context_control;
        functions.context = fixture::dispatch([] { return chain_current; });
        functions.context_create = [](CUcontext* output, unsigned int, CUdevice) {
            static unsigned int creations{};
            *output = chain_current = reinterpret_cast<CUcontext>(new int{0});
            fixture::state.configurations[*output].version = 3020 + ++creations;
            return CUDA_SUCCESS;
        };
        functions.context_destroy = [](CUcontext context) {
            expect(context == chain_current, "context-control cleanup owns the native context");
            fixture::state.configurations.erase(context);
            delete reinterpret_cast<int*>(context);
            chain_current = nullptr;
            return CUDA_SUCCESS;
        };
        functions.module_load = [](CUmodule* output, const void*) {
            *output = reinterpret_cast<CUmodule>(&chain);
            return CUDA_SUCCESS;
        };
        functions.module_unload = [](CUmodule module) {
            expect(module == reinterpret_cast<CUmodule>(&chain), "owned query fixture module");
            return CUDA_SUCCESS;
        };
        functions.function_get = [](CUfunction* output, CUmodule module, const char* name) {
            expect(module == reinterpret_cast<CUmodule>(&chain) && std::string_view(name) == "noop",
                   "owned query fixture entry");
            *output = reinterpret_cast<CUfunction>(&chain_context);
            return CUDA_SUCCESS;
        };
        functions.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
            return CUDA_ERROR_INVALID_VALUE;  // The original kernel has zero parameters.
        };
        functions.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                               unsigned int, unsigned int, unsigned int,
                                               unsigned int, CUstream, void**, void**) {
            chain.matmuls = 1;  // Existing trailing-event fixture remains pending until can_stop.
            return CUDA_SUCCESS;
        };
    }
    ExecutionService service(std::move(config), functions);
    expect(service.start(), "start actual service composition");
    auto client = std::async(std::launch::async, [&] {
        if (is_context)
            application_context_controls(metadata_driver, metadata_runtime);
        else if (is_random)
            application_random(provider, quarantine);
        else if (metadata_driver)
            application_runtime_metadata(metadata_driver, metadata_runtime);
        else if (is_chain)
            application_chain(provider, uses_blas_alias);
        else
            application(provider, quarantine);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{60};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded metadata progress");
    client.get();
    expect(native.gpu_calls == 0 && glimmer_lt_native_calls() == 0,
           "no GPU/caller-vendor calls for metadata");
    if (is_context) {
        namespace fixture = glimmer::tests::context_control;
        expect(fixture::state.configurations.empty() && fixture::state.driver_calls == 17 &&
                   fixture::state.runtime_calls == 13 && fixture::state.resets == 2,
               "all native controls ran on the owner and every context was retired");
        expect(service.request_stop(), "seal context-control service");
        while (!service.finished())
            expect(service.step(1) && Clock::now() < deadline, "bounded control shutdown");
        return;
    }
    if (is_random) {
        expect(glimmer_random_native_calls() == 0 &&
                   random_native.creates == (quarantine ? 2U : 3U) && random_native.seeds == 3 &&
                   random_native.seed == UINT64_MAX - 17 &&
                   random_native.destroys == (quarantine ? 1U : 2U),
               "exact original random calls on the service only, including once-only cleanup");
        if (!quarantine) {
            expect(service.request_stop(), "seal random service");
            while (!service.finished())
                expect(service.step(1) && Clock::now() < deadline, "bounded random shutdown");
        }
        return;
    }
    if (!quarantine) {
        expect(metadata_driver ? chain.metadata_calls.load() == 10
               : is_chain ? native.creates == 6 && native.destroys == 5 && chain.heuristics == 2 &&
                                chain.checks == 1 && chain.matmuls == 1 &&
                                chain.handle_destroys == 1 && chain.event_destroys == 1
                          : native.creates == 17004 && native.destroys == 17003,
               "all original metadata ownership retired exactly once");
        expect(service.request_stop(), "seal ordinary service cleanup");
        while (metadata_driver && !service.finished())
            expect(service.step(1) && Clock::now() < deadline,
                   "bounded owned metadata-worker shutdown");
        expect(service.finished(), "all independent metadata owners retired after stop");
    }
    // Quarantine is isolated in a separate process. Retention is not presented
    // as recoverable cleanup or leak-free unknown-ownership behavior.
}
void metadata_retirement() {
    using namespace glimmer::control;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual metadata retirement service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-lt-retirement-" + std::to_string(::getpid()) + ".sock";
    config.exit_when_idle = true;
    const auto path = config.transport.socket_path;
    auto functions = dispatch();
    functions.lt.preference_destroy = [](cublasLtMatmulPreference_t handle) {
        auto* object = reinterpret_cast<Object*>(handle);
        expect(object && object->is_live, "one original live preference destructor attempt");
        ++native.destroys;
        return CUBLAS_STATUS_INTERNAL_ERROR;  // No proof whether ownership was consumed.
    };
    auto service = std::make_unique<ExecutionService>(std::move(config), functions);
    expect(service->start(), "start actual retirement service");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *identity, 3000});
        ExecutionPayloadWriter bootstrap;
        expect(bootstrap.write_u32(kExecutionLtMetadataRole) && bootstrap.write_u32(0),
               "encode CPU-only metadata role");
        const auto opened = connection.open(bootstrap.take());
        expect(opened && opened->status == CUDA_SUCCESS, "open metadata without CUDA context");
        ExecutionPayloadWriter creation;
        expect(creation.write_u32(static_cast<std::uint32_t>(ExecutionLtObjectKind::kPreference)) &&
                   creation.write_u32(CUDA_R_32F) && creation.write_u32(CUBLAS_COMPUTE_32F) &&
                   creation.write_u32(CUDA_R_32F) && creation.write_u64(0) &&
                   creation.write_u64(0) && creation.write_u64(0),
               "encode original preference creation");
        const auto created =
            connection.exchange(ExecutionOpcode::kCreateLtMetadata, creation.take());
        expect(created && created->status == CUBLAS_STATUS_SUCCESS && created->id,
               "real service retains original preference");
        // Socket destruction disconnects with a live descriptor; no explicit
        // destroy reply can convert the native failure into a prior quarantine.
    });
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service->step(1) && Clock::now() < deadline, "bounded raw metadata progress");
    client.get();
    while (!native.destroys)
        expect(service->step(1) && Clock::now() < deadline, "observe disconnected owner cleanup");
    expect(!service->finished(), "failed native destructor cannot make service report finished");
    for (unsigned int iteration = 0; iteration != 8; ++iteration)
        expect(service->step(0) && !service->finished(),
               "quarantined metadata owner remains tracked after repeated progress");
    expect(!service->request_stop() && native.creates == 1 && native.destroys == 1 &&
               native.gpu_calls == 0 && glimmer_lt_native_calls() == 0,
           "unproven native cleanup cannot succeed, retry or enter caller/GPU SDK");
    // request_stop has unlinked the socket. This isolated process intentionally
    // retains the quarantined owner; it does not claim leak-free recovery.
    (void)service.release();
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[2]) == "--metadata-retirement") {
            metadata_retirement();
            ::_exit(0);
        }
        const bool is_context = argc == 5 && std::string_view(argv[2]) == "--context-control";
        expect(argc == 2 || is_context ||
                   (argc == 5 && std::string_view(argv[2]) == "--runtime-metadata") ||
                   (argc == 3 && (std::string_view(argv[2]) == "--quarantine" ||
                                  std::string_view(argv[2]) == "--chain" ||
                                  std::string_view(argv[2]) == "--random" ||
                                  std::string_view(argv[2]) == "--random-quarantine" ||
                                  std::string_view(argv[2]) == "--blas-chain")),
               "fixture arguments");
        const bool is_random = argc == 3 && std::string_view(argv[2]).starts_with("--random");
        const bool quarantine = argc == 3 && (std::string_view(argv[2]) == "--quarantine" ||
                                              std::string_view(argv[2]) == "--random-quarantine");
        run(argv[1], quarantine, argc == 3 && !quarantine,
            argc == 3 && std::string_view(argv[2]) == "--blas-chain", argc == 5 ? argv[3] : nullptr,
            argc == 5 ? argv[4] : nullptr, is_random, is_context);
        if (quarantine) ::_exit(0);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
