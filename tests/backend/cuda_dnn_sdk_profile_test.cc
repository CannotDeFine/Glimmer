#include "internal/cuda_dnn_registry.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
namespace profile = glimmer::internal::cudnn_profile;

static_assert(profile::kHeaderVersion == CUDNN_VERSION);
static_assert(profile::kSdkMajor == CUDNN_MAJOR);
static_assert(!profile::status_is_representable(-1));
static_assert(
    profile::status_is_representable(static_cast<std::int32_t>(profile::status_enum_max())));
static_assert(
    !profile::status_is_representable(static_cast<std::int32_t>(profile::status_enum_max() + 1)));
static_assert(static_cast<int>(profile::profile_mismatch_status()) == 1);
#if CUDNN_MAJOR == 8
static_assert(profile::kHeaderVersion == 8907);
static_assert(profile::status_enum_max() == 15);
static_assert(profile::data_type_enum_max() == 15);
static_assert(profile::behavior_note_enum_max() == 3);
static_assert(profile::host_allocation_status() == CUDNN_STATUS_ALLOC_FAILED);
static_assert(!profile::status_is_representable(4003));
#else
static_assert(profile::status_enum_max() == 8191);
static_assert(profile::data_type_enum_max() == 31);
static_assert(profile::behavior_note_enum_max() == 7);
static_assert(profile::host_allocation_status() ==
              CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED);
#endif

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void native_result(const CudaDnnResult& result, cudnnStatus_t status) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status,
           "original selected-profile status must remain unchanged");
}
void rejected(const CudaDnnResult& result, CUresult status) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == status &&
               !result.original_status && result.id == 0,
           "pre-SDK rejection must not fabricate native status or descriptor authority");
}

// This fixture deliberately does not include the SDK9 empirical metadata
// fixtures. Its callback signatures are checked against the selected headers,
// and it models only the tensor calls actually used by these profile tests.
struct Provider {
    bool is_alive{}, is_configured{};
    CudaDnnTensor tensor;
    std::size_t version{CUDNN_VERSION};
    unsigned int creates{}, destroys{}, sets{}, gets{}, backend_creates{}, backend_destroys{};
    std::uint32_t last_type{};
    std::optional<std::uint32_t> reported_type;
};
Provider provider;

cudnnTensorDescriptor_t tensor_identity() {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): inert fake SDK identity.
    return reinterpret_cast<cudnnTensorDescriptor_t>(&provider);
}
cudnnStatus_t create_tensor(cudnnTensorDescriptor_t* output) {
    expect(output && !provider.is_alive, "one live tensor identity per independent provider");
    ++provider.creates;
    provider.is_alive = true;
    *output = tensor_identity();
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t destroy_tensor(cudnnTensorDescriptor_t tensor) {
    expect(provider.is_alive && tensor == tensor_identity(), "only owned live tensor is destroyed");
    ++provider.destroys;
    provider.is_alive = false;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t set_tensor(cudnnTensorDescriptor_t tensor, cudnnDataType_t type, int rank,
                         const int* sizes, const int* strides) {
    expect(provider.is_alive && tensor == tensor_identity(), "tensor setter uses owned identity");
    ++provider.sets;
    provider.last_type = static_cast<std::uint32_t>(type);
    if (type != CUDNN_DATA_FLOAT) return CUDNN_STATUS_BAD_PARAM;
    expect(rank == 4 && sizes && strides, "profile fixture receives exact bounded tensor arrays");
    provider.tensor = {.type = static_cast<std::uint32_t>(type), .dimensions = rank};
    std::copy_n(sizes, rank, provider.tensor.sizes.begin());
    std::copy_n(strides, rank, provider.tensor.strides.begin());
    provider.is_configured = true;
    return CUDNN_STATUS_SUCCESS;
}
// The selected SDK ABI fixes the adjacent rank/dimension/stride pointer order.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
cudnnStatus_t get_tensor(cudnnTensorDescriptor_t tensor, int requested, cudnnDataType_t* type,
                         int* rank, int* sizes, int* strides) {
    expect(provider.is_alive && provider.is_configured && tensor == tensor_identity(),
           "tensor query uses configured owned identity");
    ++provider.gets;
    const auto bits = provider.reported_type.value_or(provider.tensor.type);
    static_assert(sizeof(*type) == sizeof(bits));
    // Malformed native output is copied as bytes, never evaluated as an enum.
    std::memcpy(type, &bits, sizeof(bits));
    *rank = provider.tensor.dimensions;
    const auto count = std::min(requested, *rank);
    std::copy_n(provider.tensor.sizes.begin(), count, sizes);
    std::copy_n(provider.tensor.strides.begin(), count, strides);
    return CUDNN_STATUS_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
cudnnStatus_t unexpected_call() {
    throw std::runtime_error("unexpected SDK entry in isolated profile fixture");
}
CudaDnnDispatch dispatch() {
    CudaDnnDispatch sdk;
    sdk.tensor_create = create_tensor;
    sdk.tensor_destroy = destroy_tensor;
    sdk.tensor_set_nd = set_tensor;
    sdk.tensor_get_nd = get_tensor;
    sdk.tensor_set_4d = [](cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, int,
                           int, int) { return unexpected_call(); };
    sdk.tensor_set_4d_ex = [](cudnnTensorDescriptor_t, cudnnDataType_t, int, int, int, int, int,
                              int, int, int) { return unexpected_call(); };
    sdk.tensor_set_nd_ex = [](cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int,
                              const int*) { return unexpected_call(); };
    sdk.tensor_get_4d = [](cudnnTensorDescriptor_t, cudnnDataType_t*, int*, int*, int*, int*, int*,
                           int*, int*, int*) { return unexpected_call(); };
    sdk.tensor_size = [](cudnnTensorDescriptor_t, std::size_t*) { return unexpected_call(); };
    sdk.activation_create = [](CudaDnnActivation*) { return unexpected_call(); };
    sdk.activation_destroy = [](CudaDnnActivation) { return unexpected_call(); };
    sdk.activation_set = [](CudaDnnActivation, cudnnActivationMode_t, cudnnNanPropagation_t,
                            double) { return unexpected_call(); };
    sdk.activation_get = [](CudaDnnActivation, cudnnActivationMode_t*, cudnnNanPropagation_t*,
                            double*) { return unexpected_call(); };
    sdk.version = [] { return provider.version; };
    sdk.backend_create = [](cudnnBackendDescriptorType_t, cudnnBackendDescriptor_t* output) {
        ++provider.backend_creates;
        expect(output && !*output, "failed backend creation cannot publish native identity");
        return CUDNN_STATUS_BAD_PARAM;
    };
    sdk.backend_destroy = [](cudnnBackendDescriptor_t) {
        ++provider.backend_destroys;
        return unexpected_call();
    };
    return sdk;
}
CudaDnnTensor input_tensor() {
    return {.type = CUDNN_DATA_FLOAT,
            .dimensions = 4,
            .sizes = {1, 2, 3, 4},
            .strides = {24, 12, 4, 1}};
}
void trait_domains() {
    expect(profile::known_metadata_rejection(CUDNN_STATUS_BAD_PARAM) &&
               profile::known_metadata_rejection(CUDNN_STATUS_NOT_SUPPORTED) &&
               !profile::known_metadata_rejection(CUDNN_STATUS_SUCCESS) &&
               !profile::known_metadata_rejection(CUDNN_STATUS_INTERNAL_ERROR) &&
               !profile::known_metadata_rejection(profile::host_allocation_status()),
           "selected-profile rejection classes do not reinterpret native error numbers");
    expect(profile::supports_backend_kind(17) && profile::supports_backend_kind(33) &&
               !profile::supports_backend_kind(41),
           "backend descriptor kind support follows the selected headers");
    expect(profile::supports_backend_kind(38) == (CUDNN_MAJOR == 9),
           "SDK8 cannot acquire SDK9 device-property descriptor authority");
    for (const auto name :
         std::array<std::uint32_t, 9>{204, 303, 304, 407, 803, 1307, 2700, 2701, 2702})
        expect(profile::supports_backend_attribute(name) == (CUDNN_MAJOR == 9),
               "SDK9-only attribute schema is absent from SDK8");
    for (const auto name : std::array<std::uint32_t, 4>{900, 901, 909, 1524})
        expect(profile::supports_backend_attribute(name),
               "common attribute declarations remain available without qualifying behavior");
}
void tensor_conversion_and_original_errors() {
    provider = {};
    CudaDnnRegistry registry(dispatch());
    const auto created = registry.create(CudaDnnObjectKind::kTensor);
    native_result(created, CUDNN_STATUS_SUCCESS);
    expect(created.id != 0 && provider.creates == 1, "native creation grants one owned identity");
    auto input = input_tensor();
    native_result(registry.set_tensor(created.id, CudaDnnTensorSetter::kNd, input),
                  CUDNN_STATUS_SUCCESS);
    input.type = profile::data_type_enum_max();
    native_result(registry.set_tensor(created.id, CudaDnnTensorSetter::kNd, input),
                  CUDNN_STATUS_BAD_PARAM);
    expect(provider.last_type == input.type && provider.sets == 2 &&
               !registry.snapshot().is_quarantined,
           "representable reserved type reaches SDK and its known rejection retains metadata");
    ++input.type;
    rejected(registry.set_tensor(created.id, CudaDnnTensorSetter::kNd, input),
             CUDA_ERROR_INVALID_VALUE);
    expect(provider.sets == 2, "out-of-profile type never forms an SDK enum or enters its setter");
    const auto observed = registry.get_tensor(created.id, 4);
    native_result(observed, CUDNN_STATUS_SUCCESS);
    const auto expected = input_tensor();
    expect(observed.tensor.type == expected.type && observed.tensor.sizes == expected.sizes &&
               observed.tensor.strides == expected.strides,
           "known native rejection preserves independently queried prior tensor metadata");
    native_result(registry.destroy(created.id), CUDNN_STATUS_SUCCESS);
    native_result(registry.shutdown(), CUDNN_STATUS_SUCCESS);
    expect(provider.destroys == 1 && !provider.is_alive && registry.snapshot().is_closed,
           "healthy selected-profile descriptor retires exactly once");
}
void empirical_backend_gate() {
    provider = {};
    CudaDnnRegistry registry(dispatch());
    // A header match alone does not qualify the historical 9.10.2 byte-footprint
    // evidence for SDK8 or for another major9 patch release.
    provider.version = 8907;
    rejected(registry.create_backend(17), CUDA_ERROR_NOT_SUPPORTED);
    provider.version = 91003;
    rejected(registry.create_backend(17), CUDA_ERROR_NOT_SUPPORTED);
    expect(provider.backend_creates == 0 && registry.snapshot().objects == 0,
           "unqualified native versions cannot reach empirical backend constructors");
    provider.version = 91002;
    const auto property = registry.create_backend(38);
#if CUDNN_MAJOR == 8
    rejected(property, CUDA_ERROR_NOT_SUPPORTED);
    expect(provider.backend_creates == 0,
           "even an injected version cannot bypass SDK8 descriptor enum/schema limits");
#else
    native_result(property, CUDNN_STATUS_BAD_PARAM);
    expect(provider.backend_creates == 1 && !property.id && !registry.snapshot().is_quarantined,
           "qualified SDK9 constructor retains original rejected status without granting identity");
#endif
    native_result(registry.shutdown(), CUDNN_STATUS_SUCCESS);
    expect(provider.backend_destroys == 0 && registry.snapshot().is_closed,
           "rejected backend construction invents no cleanup obligation");
}
void malformed_native_type() {
    provider = {};
    CudaDnnRegistry registry(dispatch());
    const auto created = registry.create(CudaDnnObjectKind::kTensor);
    native_result(created, CUDNN_STATUS_SUCCESS);
    native_result(registry.set_tensor(created.id, CudaDnnTensorSetter::kNd, input_tensor()),
                  CUDNN_STATUS_SUCCESS);
    provider.reported_type = profile::data_type_enum_max() + 1;
    const auto result = registry.get_tensor(created.id, 4);
    expect(result.disposition == CudaDnnDisposition::kRejected &&
               result.status == CUDA_ERROR_UNKNOWN &&
               result.original_status == CUDNN_STATUS_SUCCESS && result.tensor.dimensions == 0 &&
               registry.snapshot().is_quarantined,
           "malformed native enum output retains provenance but publishes no usable metadata");
    rejected(registry.destroy(created.id), CUDA_ERROR_UNKNOWN);
    rejected(registry.shutdown(), CUDA_ERROR_UNKNOWN);
    expect(provider.destroys == 0 && provider.is_alive && registry.snapshot().objects == 1,
           "uncertain output quarantines resource instead of issuing speculative cleanup");
    // The negative proof intentionally retains uncertain native ownership.
    // Exit this isolated child without destroying or leaking into the test host.
    ::_exit(0);
}
void isolated_malformed_native_type() {
    const auto child = ::fork();
    expect(child >= 0, "fork isolates deliberate ownership quarantine");
    if (!child) {
        try {
            malformed_native_type();
        } catch (...) {
            ::_exit(1);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "malformed native output child must finish with every quarantine assertion satisfied");
}
}  // namespace

int main() {
    try {
        trait_domains();
        tensor_conversion_and_original_errors();
        empirical_backend_gate();
        isolated_malformed_native_type();
        std::cout << "cuDNN SDK profile " << CUDNN_VERSION << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
