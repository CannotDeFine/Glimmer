#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Ordinary installed-SDK input, never an automatic GPU CTest. Complete hex
// snapshots support a later native/owned byte comparison under matched runtime,
// SDK and device identities. DEFAULT's vendor permutation is not emulated here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
bool cleanup_failed{};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status) != 0)
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (release() != 0) cleanup_failed = true;
    }
    Handle* output() {
        return &value_;
    }
    Handle get() const {
        return value_;
    }
    void close() {
        checked(release(), "native cleanup");
    }

   private:
    int release() noexcept {
        if (!value_) return 0;
        return static_cast<int>(Destroy(std::exchange(value_, Handle{})));
    }
    Handle value_{};
};
using Filter = Resource<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor>;
using Memory = Resource<CUdeviceptr, cuMemFree>;
constexpr std::size_t kGuard = 256;
constexpr std::size_t kMaximumBufferBytes = 1024 * 1024;
struct Buffer {
    Buffer(std::size_t count, unsigned int seed) : bytes(count), initial(count + 2 * kGuard) {
        expect(count > 0 && count <= kMaximumBufferBytes, "bounded nonempty native buffer size");
        for (std::size_t i = 0; i != initial.size(); ++i)
            initial[i] = static_cast<std::byte>((i * 37 + (i >> 8) * 13 + seed) & 0xff);
        checked(cuMemAlloc(memory.output(), initial.size()), "allocate complete guarded storage");
        reset();
    }
    void reset() const {
        checked(cuMemcpyHtoD(memory.get(), initial.data(), initial.size()),
                "initialize complete guarded storage");
    }
    void* pointer() const {
        return reinterpret_cast<void*>(memory.get() + kGuard);
    }
    std::vector<std::byte> snapshot() const {
        std::vector<std::byte> observed(initial.size());
        checked(cuMemcpyDtoH(observed.data(), memory.get(), observed.size()),
                "snapshot complete allocation");
        return observed;
    }
    void guards(const std::vector<std::byte>& observed) const {
        expect(observed.size() == initial.size() &&
                   std::equal(initial.begin(), initial.begin() + kGuard, observed.begin()) &&
                   std::equal(initial.end() - kGuard, initial.end(), observed.end() - kGuard),
               "every prefix and suffix guard byte remains unchanged");
    }
    std::size_t bytes;
    std::vector<std::byte> initial;
    Memory memory;
};
void hex(std::string_view name, const std::vector<std::byte>& bytes) {
    constexpr std::string_view digits = "0123456789abcdef";
    std::cout << name << ':' << bytes.size() << ':';
    for (const auto byte : bytes) {
        const auto value = std::to_integer<unsigned int>(byte);
        std::cout.put(digits[value >> 4]);
        std::cout.put(digits[value & 15]);
    }
    std::cout << '\n';
    expect(static_cast<bool>(std::cout), "complete byte evidence is written");
}
bool equal_payload(const std::vector<std::byte>& left, const std::vector<std::byte>& right) {
    return left.size() == right.size() &&
           std::equal(left.begin() + kGuard, left.end() - kGuard, right.begin() + kGuard);
}
struct Case {
    cudnnDataType_t type;
    cudnnTensorFormat_t format;
    cudnnReorderType_t reorder;
    std::array<int, 4> shape;
    std::size_t bias_element_bytes;
};
constexpr std::array cases{
    Case{CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DEFAULT_REORDER, {64, 32, 3, 2}, 4},
    Case{CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_NO_REORDER, {64, 32, 3, 2}, 32},
    Case{CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, CUDNN_NO_REORDER, {64, 3, 2, 2}, 4},
    Case{CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, CUDNN_NO_REORDER, {64, 3, 2, 2}, 8}};

unsigned int run(cudnnHandle_t handle, CUstream stream, cudnnFilterDescriptor_t filter) {
    unsigned int calls{};
    for (std::size_t index = 0; index != cases.size(); ++index) {
        const auto& test = cases[index];
        checked(cudnnSetFilterNdDescriptor(filter, test.type, test.format, 4, test.shape.data()),
                "set original four-dimensional filter");
        // The configured SDK's vectorized Filter4d getter expands C differently;
        // use the authoritative Nd getter and original byte-size query.
        cudnnDataType_t type{};
        cudnnTensorFormat_t format{};
        int rank{};
        std::array<int, 8> shape{};
        shape.fill(0x12345678);
        checked(cudnnGetFilterNdDescriptor(filter, 8, &type, &format, &rank, shape.data()),
                "query original complete Nd metadata");
        expect(type == test.type && format == test.format && rank == 4 &&
                   std::equal(test.shape.begin(), test.shape.end(), shape.begin()) &&
                   std::all_of(shape.begin() + 4, shape.end(),
                               [](int value) { return value == 0x12345678; }),
               "complete logical filter metadata and untouched getter tail match");
        std::size_t bytes{};
        checked(cudnnGetFilterSizeInBytes(filter, &bytes), "query original filter byte extent");
        expect(bytes > 0 && bytes <= kMaximumBufferBytes, "bounded original filter byte extent");
        const auto bias_bytes = static_cast<std::size_t>(test.shape[0]) * test.bias_element_bytes;
        Buffer input(bytes, 0x26), output(bytes, 0xbd), bias(bias_bytes, 0x53),
            reordered_bias(bias_bytes, 0xc9);
        for (const int flag :
             {0, 1, -1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
            std::vector<std::byte> first_filter, first_bias;
            for (unsigned int repetition = 0; repetition != 2; ++repetition) {
                output.reset();
                reordered_bias.reset();
                // Flag zero must not inspect either arbitrary caller address.
                // Nonzero signed flags retain valid, complete device storage.
                const void* bias_input = flag ? bias.pointer() : reinterpret_cast<const void*>(1);
                void* bias_output = flag ? reordered_bias.pointer() : reinterpret_cast<void*>(3);
                checked(cudnnReorderFilterAndBias(handle, filter, test.reorder, input.pointer(),
                                                  output.pointer(), flag, bias_input, bias_output),
                        "execute original filter/bias reorder");
                ++calls;
                checked(cuStreamSynchronize(stream), "complete original reorder stream");
                const auto observed_input = input.snapshot(), observed_filter = output.snapshot(),
                           observed_bias = bias.snapshot(),
                           observed_reordered_bias = reordered_bias.snapshot();
                expect(observed_input == input.initial && observed_bias == bias.initial,
                       "all source payload and guard bytes remain unchanged");
                output.guards(observed_filter);
                reordered_bias.guards(observed_reordered_bias);
                expect(!equal_payload(observed_filter, output.initial),
                       "successful filter call changes its destination payload");
                if (test.reorder == CUDNN_NO_REORDER) {
                    expect(equal_payload(observed_filter, input.initial),
                           "NO_REORDER copies the complete authoritative filter byte span");
                    if (flag)
                        expect(equal_payload(observed_reordered_bias, bias.initial),
                               "NO_REORDER copies every byte of the mode-specific bias extent");
                }
                if (!flag)
                    expect(observed_reordered_bias == reordered_bias.initial,
                           "ignored bias destination is completely unchanged");
                else
                    expect(!equal_payload(observed_reordered_bias, reordered_bias.initial),
                           "every nonzero signed flag actually changes the bias destination");
                if (!repetition) {
                    first_filter = observed_filter;
                    first_bias = observed_reordered_bias;
                } else {
                    expect(first_filter == observed_filter && first_bias == observed_reordered_bias,
                           "repeated original calls have identical complete byte outputs");
                }
                std::cout << "case=" << index << " type=" << static_cast<int>(type)
                          << " format=" << static_cast<int>(format)
                          << " reorder=" << static_cast<int>(test.reorder) << " bias_flag=" << flag
                          << " repetition=" << repetition << " filter_bytes=" << bytes
                          << " bias_bytes=" << bias_bytes << " status=0\n";
                hex("filter_input", observed_input);
                hex("filter_output", observed_filter);
                hex("bias_input", observed_bias);
                hex("bias_output", observed_reordered_bias);
            }
        }
        reordered_bias.memory.close();
        bias.memory.close();
        output.memory.close();
        input.memory.close();
    }
    return calls;
}
}  // namespace
int main() {
    try {
        Filter filter;
        checked(cudnnCreateFilterDescriptor(filter.output()), "create metadata before CUDA");
        checked(cuInit(0), "initialize Driver");
        CUdevice device{};
        checked(cuDeviceGet(&device, 0), "select device");
        Resource<CUcontext, cuCtxDestroy> context;
        checked(cuCtxCreate(context.output(), 0, device), "create context");
        Resource<CUstream, cuStreamDestroy> stream;
        checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING), "create explicit stream");
        Resource<cudnnHandle_t, cudnnDestroy> handle;
        checked(cudnnCreate(handle.output()), "create original cuDNN handle");
        checked(cudnnSetStream(handle.get(), reinterpret_cast<cudaStream_t>(stream.get())),
                "bind explicit original stream");
        std::cout << "reorder-byte-snapshot-v1\n";
        const auto calls = run(handle.get(), stream.get(), filter.get());
        expect(calls == 40, "complete four-case five-flag two-repetition matrix");
        checked(cudnnSetStream(handle.get(), nullptr), "unbind stream");
        handle.close();
        stream.close();
        context.close();
        filter.close();
        expect(!cleanup_failed, "every native cleanup succeeded exactly once");
        std::cout << "complete calls=" << calls
                  << " default_permutation_oracle=native_owned_byte_comparison_required"
                     " full_device_qualified=false priority_qualified=false\n";
        expect(static_cast<bool>(std::cout), "complete success receipt is written");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
