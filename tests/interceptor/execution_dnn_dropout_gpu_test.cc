#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Ordinary installed-SDK input only. It has no injected provider, automatic
// GPU CTest, timing, full-device, dropout-layer or persistent-kernel claim.
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
using Dropout = Resource<cudnnDropoutDescriptor_t, cudnnDestroyDropoutDescriptor>;
using Memory = Resource<CUdeviceptr, cuMemFree>;
constexpr std::size_t kGuard = 256;
constexpr std::size_t kMaximumStateBytes = 64 * 1024 * 1024;

struct States {
    explicit States(std::size_t count) : bytes(count), initial(count + 2 * kGuard) {
        for (std::size_t i = 0; i != initial.size(); ++i)
            initial[i] = static_cast<std::byte>((i * 37 + 0x96) & 0xff);
        checked(cuMemAlloc(memory.output(), initial.size()), "allocate guarded RNG states");
        checked(cuMemcpyHtoD(memory.get(), initial.data(), initial.size()),
                "initialize complete guarded RNG storage");
    }
    CUdeviceptr address() const {
        return memory.get() + kGuard;
    }
    void* pointer() const {
        return reinterpret_cast<void*>(address());
    }
    std::vector<std::byte> snapshot() const {
        std::vector<std::byte> observed(initial.size());
        checked(cuMemcpyDtoH(observed.data(), memory.get(), observed.size()),
                "read complete guarded RNG storage");
        return observed;
    }
    void check_guards(const std::vector<std::byte>& observed) const {
        expect(observed.size() == initial.size(), "complete state snapshot extent");
        expect(std::equal(initial.begin(), initial.begin() + kGuard, observed.begin()) &&
                   std::equal(initial.end() - kGuard, initial.end(), observed.end() - kGuard),
               "every prefix and suffix guard byte is unchanged");
    }
    std::size_t bytes;
    std::vector<std::byte> initial;
    Memory memory;
};

struct Evidence {
    unsigned sets{}, restores{}, queries{};
};
void run(cudnnHandle_t handle, cudnnDropoutDescriptor_t initialized,
         cudnnDropoutDescriptor_t restored, Evidence& evidence) {
    std::size_t bytes{};
    checked(cudnnDropoutGetStatesSize(handle, &bytes), "query original RNG state size");
    ++evidence.queries;
    expect(bytes > 0 && bytes <= kMaximumStateBytes, "bounded nonempty RNG state allocation");
    States source(bytes), copy(bytes);
    checked(
        cudnnSetDropoutDescriptor(initialized, handle, 0.125F, nullptr, 0, 0xffff000012345678ULL),
        "null Set before initialization");
    ++evidence.sets;
    checked(cuCtxSynchronize(), "complete initial null configuration");
    expect(source.snapshot() == source.initial && copy.snapshot() == copy.initial,
           "null initialization does not fabricate or write RNG storage");

    constexpr std::array seeds{0x12345678abcdef01ULL, 0xfedcba9800000001ULL};
    for (const auto seed : seeds) {
        const auto before = source.snapshot();
        checked(
            cudnnSetDropoutDescriptor(initialized, handle, 0.25F, source.pointer(), bytes, seed),
            "initialize original device RNG states");
        ++evidence.sets;
        std::size_t queried{};
        checked(cudnnDropoutGetStatesSize(handle, &queried),
                "size query after queued initialization");
        ++evidence.queries;
        expect(queried == bytes, "state size remains stable for the same native handle");
        // Initialization can use opaque native streams; synchronize this owned
        // context rather than assuming the handle stream proves completion.
        checked(cuCtxSynchronize(), "complete original RNG initialization");
        const auto initialized_bytes = source.snapshot();
        source.check_guards(initialized_bytes);
        expect(!std::equal(initialized_bytes.begin() + kGuard, initialized_bytes.end() - kGuard,
                           before.begin() + kGuard),
               "nonnull Set actually modifies the RNG state payload");
        checked(cudnnSetDropoutDescriptor(initialized, handle, 0.625F, nullptr, bytes, seed ^ 1ULL),
                "null Set updates probability without reinitialization");
        ++evidence.sets;
        checked(cuCtxSynchronize(), "complete probability-only update");
        expect(source.snapshot() == initialized_bytes,
               "null Set preserves every byte of the previously initialized state");

        checked(cuMemcpyDtoD(copy.address(), source.address(), bytes),
                "copy actual initialized native states without interpreting opaque bytes");
        const auto copied = copy.snapshot();
        copy.check_guards(copied);
        expect(std::equal(copied.begin() + kGuard, copied.end() - kGuard,
                          initialized_bytes.begin() + kGuard),
               "complete native state payload is copied to distinct application storage");
        for (const float probability : {0.375F, 0.75F}) {
            checked(cudnnRestoreDropoutDescriptor(restored, handle, probability, copy.pointer(),
                                                  bytes, ~seed),
                    "restore original states with a different, ignored seed");
            ++evidence.restores;
            checked(cuCtxSynchronize(), "complete state restoration");
            expect(copy.snapshot() == copied && source.snapshot() == initialized_bytes,
                   "Restore neither reinitializes copied states nor mutates source storage");
        }
    }
    // Application-owned storage may be released after physical quiescence
    // before either descriptor is destroyed. A CPU binding is not ownership.
    checked(cuCtxSynchronize(), "quiesce before application releases its state buffers");
    copy.memory.close();
    source.memory.close();
}
}  // namespace

int main() {
    try {
        Dropout initialized, restored;
        checked(cudnnCreateDropoutDescriptor(initialized.output()), "create metadata before CUDA");
        checked(cudnnCreateDropoutDescriptor(restored.output()), "create second CPU descriptor");
        checked(cuInit(0), "initialize Driver");
        CUdevice device{};
        checked(cuDeviceGet(&device, 0), "select device");
        Resource<CUcontext, cuCtxDestroy> context;
        checked(cuCtxCreate(context.output(), 0, device), "create context");
        Resource<CUstream, cuStreamDestroy> stream;
        checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING), "create stream");
        Resource<cudnnHandle_t, cudnnDestroy> handle;
        checked(cudnnCreate(handle.output()), "create cuDNN handle");
        checked(cudnnSetStream(handle.get(), reinterpret_cast<cudaStream_t>(stream.get())),
                "bind explicit stream");
        Evidence evidence;
        run(handle.get(), initialized.get(), restored.get(), evidence);
        expect(evidence.sets == 5 && evidence.restores == 4 && evidence.queries == 3,
               "complete initialization/null-update/restore/query matrix");
        checked(cudnnSetStream(handle.get(), nullptr), "unbind stream");
        handle.close();
        stream.close();
        context.close();
        restored.close();
        initialized.close();
        expect(!cleanup_failed, "checked once-only cleanup after context and storage teardown");
        std::cout << "cuDNN dropout state input passed: set_calls=" << evidence.sets
                  << "; restore_calls=" << evidence.restores << "; queries=" << evidence.queries
                  << "; dropout_layer_qualified=false; full_device_qualified=false"
                     "; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
