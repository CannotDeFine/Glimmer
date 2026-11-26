#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Unchanged ordinary Driver input for native/profile/readonly-sharing controls.
// Never an automatic GPU CTest or a utilization/performance qualification.
namespace {
constexpr unsigned int kBlocks = 1024, kThreads = 256, kCalls = 4, kGuard = 64;
constexpr std::size_t kValues = std::size_t{kBlocks} * kThreads;
constexpr std::uint32_t kMarker = 0xa9c5e137U;
bool has_failure{};
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}
template <typename Value, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (value_ && !has_failure && Destroy(value_) != CUDA_SUCCESS) has_failure = true;
    }
    Value get() const {
        return value_;
    }
    Value* output() {
        return &value_;
    }

   private:
    Value value_{};
};
std::uint64_t checksum(const std::vector<std::uint32_t>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : values)
        for (unsigned int shift = 0; shift < 32; shift += 8) {
            hash ^= (value >> shift) & 255U;
            hash *= 1099511628211ULL;
        }
    return hash;
}
constexpr char kSource[] = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry sharing_compute(.param .u64 output, .param .u32 count, .param .u32 seed) {
 .reg .pred %p<3>;
 .reg .b32 %r<12>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output];
 ld.param.u32 %r1, [count]; ld.param.u32 %r2, [seed];
 // Logical geometry is part of the oracle, not just a capacity assumption.
 mov.u32 %r10, %ntid.x; setp.ne.u32 %p1, %r10, 256; @%p1 bra DONE;
 mov.u32 %r10, %ntid.y; setp.ne.u32 %p1, %r10, 1; @%p1 bra DONE;
 mov.u32 %r10, %ntid.z; setp.ne.u32 %p1, %r10, 1; @%p1 bra DONE;
 mov.u32 %r10, %nctaid.x; setp.ne.u32 %p1, %r10, 1024; @%p1 bra DONE;
 mov.u32 %r10, %nctaid.y; setp.ne.u32 %p1, %r10, 1; @%p1 bra DONE;
 mov.u32 %r10, %nctaid.z; setp.ne.u32 %p1, %r10, 1; @%p1 bra DONE;
 mov.u32 %r3, %ctaid.x; mov.u32 %r4, %ntid.x; mov.u32 %r5, %tid.x;
 mad.lo.u32 %r6, %r3, %r4, %r5;
 setp.ge.u32 %p1, %r6, %r1; @%p1 bra DONE;
 mul.wide.u32 %rd2, %r6, 4; add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r7, [%rd3]; mov.u32 %r8, 64;
 add.u32 %r9, %r2, 1013904223;
LOOP:
 mul.lo.u32 %r7, %r7, 1664525; add.u32 %r7, %r7, %r9;
 sub.u32 %r8, %r8, 1; setp.ne.u32 %p2, %r8, 0; @%p2 bra LOOP;
 st.global.u32 [%rd3], %r7;
DONE: ret;
}
)ptx";

void run() {
    checked(cuInit(0), "initialize original Driver");
    CUdevice device{};
    checked(cuDeviceGet(&device, 0), "select unchanged device");
    Resource<CUcontext, cuCtxDestroy> context;
    checked(cuCtxCreate(context.output(), 0, device), "create original context");
    Resource<CUstream, cuStreamDestroy> stream;
    checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING), "create original stream");
    Resource<CUmodule, cuModuleUnload> module;
    checked(cuModuleLoadData(module.output(), kSource), "load unchanged original PTX");
    CUfunction function{};
    checked(cuModuleGetFunction(&function, module.get(), "sharing_compute"),
            "resolve original function");
    Resource<CUdeviceptr, cuMemFree> allocation;
    std::vector<std::uint32_t> expected(kValues + std::size_t{2} * kGuard, kMarker);
    for (std::size_t index = 0; index < kValues; ++index)
        expected[index + kGuard] = 0x12345678U + static_cast<std::uint32_t>(index) * 17U;
    checked(cuMemAlloc(allocation.output(), expected.size() * sizeof(expected[0])),
            "allocate full output and guards");
    checked(cuMemcpyHtoD(allocation.get(), expected.data(), expected.size() * sizeof(expected[0])),
            "upload complete initial state");
    checked(cuCtxSynchronize(), "order initial state before nonblocking execution");
    auto output = allocation.get() + kGuard * sizeof(expected[0]);
    auto count = static_cast<unsigned int>(kValues);
    for (unsigned int call = 0; call < kCalls; ++call) {
        unsigned int seed = 19U + call * 23U;
        std::array<void*, 3> parameters{&output, &count, &seed};
        checked(cuLaunchKernel(function, kBlocks, 1, 1, kThreads, 1, 1, 0, stream.get(),
                               parameters.data(), nullptr),
                "issue exactly one unchanged logical computation");
        checked(cuStreamSynchronize(stream.get()), "prove this exact logical call complete");
        expect(parameters == std::array<void*, 3>{&output, &count, &seed} &&
                   output == allocation.get() + kGuard * sizeof(expected[0]) && count == kValues &&
                   seed == 19U + call * 23U,
               "public argument slots and their values remain unchanged");
        for (std::size_t index = kGuard; index < kGuard + kValues; ++index)
            for (unsigned int iteration = 0; iteration < 64; ++iteration)
                expected[index] = expected[index] * 1664525U + 1013904223U + seed;
        std::vector<std::uint32_t> actual(expected.size());
        checked(cuMemcpyDtoH(actual.data(), allocation.get(), actual.size() * sizeof(actual[0])),
                "read complete output and both guards");
        expect(actual == expected,
               "every output and guard matches the independent exactly-once recurrence");
        std::cout << "sharing_compute_case;call=" << call << ";values=" << kValues
                  << ";guards=" << 2 * kGuard << ";checksum=" << checksum(actual) << '\n';
    }
}
}  // namespace

int main() {
    try {
        run();
        expect(!has_failure, "all original resources cleaned up normally");
        std::cout << "sharing_compute_success;logical_calls=4;replays=0;"
                     "full_device_qualified=0;performance_qualified=0\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        has_failure = true;
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
