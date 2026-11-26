#include <cuda.h>
#include <cudnn.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

// Ordinary SDK input: compile only, never an automatic GPU CTest. These queries
// require a real CUDA context and may cause vendor device/loader preparation.
// No RNN/attention computation or tensor/workspace allocation is performed.
// Failed native queries may partially write outputs. Print those complete
// snapshots diagnostically; do not require them to match owned no-publication.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
bool has_failure{};
unsigned int query_calls{}, clip_calls{};
void expect(bool condition, const char* message) {
    if (!condition) {
        has_failure = true;
        throw std::runtime_error(message);
    }
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status) != 0) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (value_ && !has_failure && static_cast<int>(Destroy(std::exchange(value_, {}))) != 0)
            has_failure = true;
    }
    Handle* output() {
        return &value_;
    }
    Handle get() const {
        return value_;
    }
    void close() {
        checked(Destroy(std::exchange(value_, {})), "release original resource");
    }

   private:
    Handle value_{};
};
using Rnn = Resource<cudnnRNNDescriptor_t, cudnnDestroyRNNDescriptor>;
using Input = Resource<cudnnRNNDataDescriptor_t, cudnnDestroyRNNDataDescriptor>;
using Attention = Resource<cudnnAttnDescriptor_t, cudnnDestroyAttnDescriptor>;
struct Descriptors {
    Rnn rnn;
    Input input;
    Attention attention;
    void initialize(unsigned int configuration) {
        const int width = configuration ? 8 : 4;
        const int sequence = configuration ? 4 : 3;
        const int batch = configuration ? 3 : 2;
        checked(cudnnCreateRNNDescriptor(rnn.output()), "create RNN descriptor");
        checked(cudnnCreateRNNDataDescriptor(input.output()), "create RNNData descriptor");
        checked(cudnnCreateAttnDescriptor(attention.output()), "create attention descriptor");
        checked(cudnnSetRNNDescriptor_v8(
                    rnn.get(), CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS,
                    CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                    CUDNN_DEFAULT_MATH, width, width, width, configuration ? 2 : 1, nullptr, 0),
                "set bounded LSTM configuration");
        const std::array lengths{sequence, sequence - 1, sequence - 2};
        float padding{};
        checked(cudnnSetRNNDataDescriptor(input.get(), CUDNN_DATA_FLOAT,
                                          CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, sequence, batch,
                                          width, lengths.data(), &padding),
                "set positive initialized RNNData for safe workspace division");
        checked(
            cudnnSetAttnDescriptor(attention.get(), 0, 1, 1.0, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                   CUDNN_DEFAULT_MATH, nullptr, nullptr, width, width, width, width,
                                   width, width, width, sequence, sequence, batch, 1),
            "set bounded attention configuration");
        const std::array<std::array<double, 2>, 4> bounds{
            {{-0.0, 0.0},
             {-1.25, 2.5},
             {-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()},
             {0.0, 0.0}}};
        for (std::size_t index = 0; index != bounds.size(); ++index) {
            const auto mode =
                index == 0 || index == 3 ? CUDNN_RNN_CLIP_NONE : CUDNN_RNN_CLIP_MINMAX;
            const auto status = cudnnRNNSetClip_v8(rnn.get(), mode, CUDNN_NOT_PROPAGATE_NAN,
                                                   bounds[index][0], bounds[index][1]);
            ++clip_calls;
            std::cout << "rnn_clip_case=" << clip_calls << "; configuration=" << configuration
                      << "; mode=" << static_cast<int>(mode) << "; lower_bits=" << std::hex
                      << std::bit_cast<std::uint64_t>(bounds[index][0])
                      << "; upper_bits=" << std::bit_cast<std::uint64_t>(bounds[index][1])
                      << std::dec << "; status=" << static_cast<int>(status) << '\n';
            checked(status, "set original clipping metadata");
        }
    }
    void close() {
        attention.close();
        input.close();
        rnn.close();
    }
};
enum class Query : unsigned int { kWeight = 1, kTemp, kAttention };
using GuardedSize = std::array<std::size_t, 3>;
constexpr GuardedSize kInitial{static_cast<std::size_t>(UINT64_C(0xa1b2c3d4e5f60718)),
                               static_cast<std::size_t>(UINT64_C(0xfedcba9876543210)),
                               static_cast<std::size_t>(UINT64_C(0x1928374655aabbcc))};
struct Observation {
    cudnnStatus_t status{};
    std::array<GuardedSize, 3> storage{kInitial, kInitial, kInitial};
};
using Aliases = std::array<std::size_t, 3>;
constexpr Aliases kSeparate{0, 1, 2};
Observation observe(cudnnHandle_t handle, const Descriptors& descriptors, Query query,
                    unsigned int mode, unsigned int mask, Aliases aliases,
                    unsigned int configuration, unsigned int pass) {
    Observation result;
    std::array<std::size_t*, 3> outputs{};
    std::array<bool, 3> used{};
    for (std::size_t i = 0; i != outputs.size(); ++i)
        if (mask & (1U << i)) {
            outputs[i] = &result.storage[aliases[i]][1];
            used[aliases[i]] = true;
        }
    if (query == Query::kWeight)
        result.status = cudnnGetRNNWeightSpaceSize(handle, descriptors.rnn.get(), outputs[0]);
    else if (query == Query::kTemp)
        result.status = cudnnGetRNNTempSpaceSizes(handle, descriptors.rnn.get(),
                                                  static_cast<cudnnForwardMode_t>(mode),
                                                  descriptors.input.get(), outputs[1], outputs[2]);
    else
        result.status = cudnnGetMultiHeadAttnBuffers(handle, descriptors.attention.get(),
                                                     outputs[0], outputs[1], outputs[2]);
    ++query_calls;
    std::cout << "rnn_query_case=" << query_calls << "; configuration=" << configuration
              << "; pass=" << pass << "; kind=" << static_cast<unsigned int>(query)
              << "; mode=" << mode << "; mask=" << mask << "; aliases=" << aliases[0] << ','
              << aliases[1] << ',' << aliases[2] << "; status=" << static_cast<int>(result.status)
              << "; failed_output_bytes_diagnostic=" << (result.status != CUDNN_STATUS_SUCCESS)
              << "; storage=" << std::hex;
    for (const auto& slot : result.storage)
        for (auto word : slot) std::cout << static_cast<std::uint64_t>(word) << ',';
    std::cout << std::dec << '\n';
    for (std::size_t i = 0; i != used.size(); ++i) {
        expect(result.storage[i][0] == kInitial[0] && result.storage[i][2] == kInitial[2],
               "all output prefix/suffix bytes are preserved");
        if (!used[i])
            expect(result.storage[i] == kInitial, "all unreferenced output storage is untouched");
    }
    const bool missing_required = query == Query::kWeight      ? !(mask & 1U)
                                  : query == Query::kAttention ? (mask & 3U) != 3U
                                                               : false;
    expect(result.status == (missing_required ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_SUCCESS),
           "configured native nullable-query status matches the safe public contract");
    return result;
}
void compare_alias(const Observation& original, const Observation& actual, Query query,
                   unsigned int mask, const Aliases& aliases) {
    expect(actual.status == original.status, "alias query preserves original status");
    if (actual.status != CUDNN_STATUS_SUCCESS) return;
    std::array<GuardedSize, 3> wanted{kInitial, kInitial, kInitial};
    const auto order = query == Query::kAttention ? Aliases{0, 2, 1} : Aliases{0, 1, 2};
    for (auto i : order)
        if (mask & (1U << i)) wanted[aliases[i]][1] = original.storage[i][1];
    expect(actual.storage == wanted,
           "complete aliased output storage matches native successful publication order");
}
}  // namespace

int main() {
    try {
        static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));
        std::array<Descriptors, 2> descriptors;
        for (unsigned int configuration = 0; configuration != descriptors.size(); ++configuration)
            descriptors[configuration].initialize(configuration);
        checked(cuInit(0), "initialize Driver after host metadata");
        CUdevice device{};
        checked(cuDeviceGet(&device, 0), "select actual CUDA device");
        Resource<CUcontext, cuCtxDestroy> context;
        checked(cuCtxCreate(context.output(), 0, device), "create actual CUDA context");
        Resource<CUstream, cuStreamDestroy> stream;
        checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING), "create explicit stream");
        Resource<cudnnHandle_t, cudnnDestroy> handle;
        checked(cudnnCreate(handle.output()), "create context-owned original cuDNN handle");
        std::array<std::array<std::array<std::array<Observation, 8>, 2>, 3>, 2> baseline{};
        for (unsigned int pass = 0; pass != 4; ++pass) {
            checked(
                cudnnSetStream(handle.get(),
                               pass % 2 ? reinterpret_cast<cudaStream_t>(stream.get()) : nullptr),
                "alternate default and explicit stream");
            for (unsigned int configuration = 0; configuration != descriptors.size();
                 ++configuration) {
                const auto& current = descriptors[configuration];
                for (auto query : {Query::kWeight, Query::kTemp, Query::kAttention}) {
                    const unsigned int allowed = query == Query::kWeight ? 1U
                                                 : query == Query::kTemp ? 6U
                                                                         : 7U;
                    const unsigned int modes = query == Query::kTemp ? 2U : 1U;
                    for (unsigned int mode = 0; mode != modes; ++mode) {
                        for (unsigned int mask = 0; mask != 8; ++mask) {
                            if (mask & ~allowed) continue;
                            const auto value = observe(handle.get(), current, query, mode, mask,
                                                       kSeparate, configuration, pass);
                            auto& first = baseline[configuration][static_cast<unsigned int>(query) -
                                                                  1][mode][mask];
                            if (!pass) first = value;
                            expect(value.status == first.status,
                                   "query status repeats across default/explicit streams");
                            if (value.status == CUDNN_STATUS_SUCCESS)
                                expect(value.storage == first.storage,
                                       "all successful size bits repeat without narrowing");
                        }
                        if (query == Query::kWeight) continue;
                        const auto& original =
                            baseline[configuration][static_cast<unsigned int>(query) - 1][mode]
                                    [allowed];
                        const std::array<Aliases, 4> aliases{
                            {{0, 1, 1}, {0, 0, 2}, {0, 1, 0}, {0, 0, 0}}};
                        const auto alias_count = query == Query::kTemp ? 1U : 4U;
                        for (unsigned int index = 0; index != alias_count; ++index)
                            compare_alias(original,
                                          observe(handle.get(), current, query, mode, allowed,
                                                  aliases[index], configuration, pass),
                                          query, allowed, aliases[index]);
                    }
                }
            }
        }
        handle.close();
        stream.close();
        for (auto& value : descriptors) value.close();
        context.close();
        expect(query_calls == 192 && clip_calls == 8 && !has_failure,
               "complete query/alias/clip matrix and checked cleanup");
        std::cout << "cuDNN RNN/attention size input passed: query_calls=192; clip_calls=8; "
                     "configurations=2; stream_passes=4; size_output_bits=64; "
                     "failed_output_snapshots=diagnostic_only; guarded_getter_calls=0; "
                     "application_compute_calls=0; device_buffer_allocations=0; "
                     "full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
