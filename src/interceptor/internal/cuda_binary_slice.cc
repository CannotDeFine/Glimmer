#include "cuda_binary_slice.h"
#include "diagnostics.h"

#include <bit>
#include <cstring>
#include <limits>
#include <cerrno>
#include <cstdio>
#include <algorithm>

namespace glimmer::interceptor {
namespace {

class MappedGemmContinuation final : public CudaBinarySliceContinuation {
   public:
    MappedGemmContinuation(std::array<unsigned int, 3> grid, std::shared_ptr<void> resources,
                           bool should_trace, QualifiedGemmMapping mapping)
        : resources_(std::move(resources)),
          plane_{grid[0], grid[1], 1},
          plane_blocks_(static_cast<std::uint64_t>(grid[0]) * grid[1]),
          progress_{plane_blocks_ * grid[2], 0, 0},
          width_(mapping == QualifiedGemmMapping::kRange ? 8U : 1U),
          should_trace_(should_trace) {}

    CUresult prepare_launch(std::array<unsigned int, 3>& grid,
                            OwnedKernelArguments& arguments) noexcept override {
        if (state_ != State::kReady) return CUDA_ERROR_INVALID_VALUE;
        const auto batch = progress_.completed_blocks / plane_blocks_;
        if (!arguments.overwrite(0, 176, std::as_bytes(std::span{&batch, 1})))
            return CUDA_ERROR_INVALID_VALUE;
        grid = plane_;
        prepared_planes_ = static_cast<unsigned int>(std::min<std::uint64_t>(
            width_, (progress_.logical_blocks - progress_.completed_blocks) / plane_blocks_));
        grid[2] = prepared_planes_;
        ++progress_.physical_launches;
        state_ = State::kPrepared;
        return CUDA_SUCCESS;
    }
    CUresult request_retreat() noexcept override {
        return state_ == State::kComplete ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
    }
    CUresult stage_completion(CUstream) noexcept override {
        if (state_ != State::kPrepared) return CUDA_ERROR_INVALID_VALUE;
        state_ = State::kStaged;
        return CUDA_SUCCESS;
    }
    CUresult inspect_stopped(bool& is_complete) noexcept override {
        is_complete = false;
        if (state_ != State::kStaged) return CUDA_ERROR_INVALID_VALUE;
        progress_.completed_blocks += plane_blocks_ * prepared_planes_;
        is_complete = progress_.completed_blocks == progress_.logical_blocks;
        state_ = is_complete ? State::kComplete : State::kReady;
        if (should_trace_) {
            const auto saved_errno = errno;
            std::array<char, 160> message{};
            const auto written =
                std::snprintf(message.data(), message.size(),
                              "[glimmer] gemm_slice blocks=%llu completed=%llu launches=%llu\n",
                              static_cast<unsigned long long>(progress_.logical_blocks),
                              static_cast<unsigned long long>(progress_.completed_blocks),
                              static_cast<unsigned long long>(progress_.physical_launches));
            if (written > 0 && static_cast<std::size_t>(written) < message.size())
                report_diagnostic(message.data());
            errno = saved_errno;
        }
        return CUDA_SUCCESS;
    }
    CudaBinarySliceProgress progress() const noexcept override {
        return progress_;
    }

   private:
    enum class State { kReady, kPrepared, kStaged, kComplete };
    std::shared_ptr<void> resources_;
    std::array<unsigned int, 3> plane_;
    std::uint64_t plane_blocks_;
    CudaBinarySliceProgress progress_;
    unsigned int width_;
    unsigned int prepared_planes_{};
    State state_{State::kReady};
    bool should_trace_{};
};

// The original parameter is a packed vendor ABI, not a C++ struct. Decode
// through memcpy, never through an alignment- or alias-sensitive typed cast.
class GemmArguments {
   public:
    explicit GemmArguments(const void* original) {
        std::memcpy(bytes.data(), original, 176);
    }
    template <typename T>
    T get(std::size_t offset) const noexcept {
        T value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    }
    bool valid(std::array<unsigned int, 3> grid) const noexcept {
        const auto m = get<std::uint32_t>(104);
        const auto n = get<std::uint32_t>(108);
        const auto k = get<std::uint32_t>(112);
        if (!m || m > 129 || !n || n > 33 || !k || k > 2048 || !grid[2] || grid[2] > 36 ||
            grid[0] != (m + 127) / 128 || grid[1] != (n + 31) / 32 ||
            get<std::uint32_t>(128) != k || get<std::uint32_t>(132) != 1 ||
            get<std::uint32_t>(116) || get<std::uint64_t>(136) || get<std::uint64_t>(144) ||
            get<std::uint32_t>(160) || get<std::uint32_t>(164))
            return false;
        // In this exact mode-one binary, the semaphore pointer at 120 is
        // inactive: both its read/wait and write paths require mode bit 0x8.
        // Preserve its opaque bytes; never dereference or reinterpret them.
        std::array<std::uint64_t, 4> pointers{}, ends{}, strides{}, leading{};
        constexpr std::array<std::size_t, 4> pointer_offsets{0, 8, 16, 168};
        for (std::size_t index = 0; index != 4; ++index) {
            leading[index] = get<std::uint32_t>(88 + index * 4);
            const auto rows = index == 1 ? n : m;
            const auto columns = index < 2 ? k : n;
            if (leading[index] < rows || leading[index] > rows + 5) return false;
            const auto extent = leading[index] * columns;
            strides[index] = get<std::uint64_t>(56 + index * 8);
            if (strides[index] < extent || strides[index] > extent + 5) return false;
            pointers[index] = get<std::uint64_t>(pointer_offsets[index]);
            // The dimension bounds above make these multiplications bounded;
            // the externally supplied device address still needs a wrap guard.
            const auto bytes_count = ((grid[2] - 1) * strides[index] + extent) * 4;
            if (!pointers[index] || pointers[index] % 16 != 0 ||
                pointers[index] > std::numeric_limits<std::uint64_t>::max() - bytes_count)
                return false;
            ends[index] = pointers[index] + bytes_count;
        }
        if (get<std::uint64_t>(24) != leading[0] * 4 || get<std::uint64_t>(32) != leading[1] * 4 ||
            get<std::uint64_t>(40) != leading[0] * 32 || get<std::uint64_t>(48) != leading[1] * 32)
            return false;
        for (const std::size_t input : {0U, 1U, 3U}) {
            if (input == 3 && pointers[2] == pointers[3] && leading[2] == leading[3] &&
                strides[2] == strides[3])
                continue;
            if (pointers[2] < ends[input] && pointers[input] < ends[2]) return false;
        }
        return true;
    }
    alignas(8) std::array<std::byte, 184> bytes{};
};

class MappedAddContinuation final : public CudaBinarySliceContinuation {
   public:
    MappedAddContinuation(std::uint64_t blocks, std::shared_ptr<void> resources,
                          std::uint64_t width, bool should_trace)
        : resources_(std::move(resources)),
          progress_{blocks, 0, 0},
          width_(width),
          should_trace_(should_trace) {}

    CUresult prepare_launch(std::array<unsigned int, 3>& grid,
                            OwnedKernelArguments& arguments) noexcept override {
        if (state_ != State::kReady ||
            !arguments.overwrite(2, 24, std::as_bytes(std::span{&progress_.completed_blocks, 1}))) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        prepared_blocks_ = std::min(width_, progress_.logical_blocks - progress_.completed_blocks);
        grid = {static_cast<unsigned int>(prepared_blocks_), 1, 1};
        ++progress_.physical_launches;
        state_ = State::kPrepared;
        return CUDA_SUCCESS;
    }
    CUresult request_retreat() noexcept override {
        // Every range already yields at its trailing event. This does not prove
        // physical stop or permit issuing the next logical block.
        return state_ == State::kComplete ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
    }
    CUresult stage_completion(CUstream) noexcept override {
        if (state_ != State::kPrepared) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        state_ = State::kStaged;
        return CUDA_SUCCESS;
    }
    CUresult inspect_stopped(bool& is_complete) noexcept override {
        is_complete = false;
        if (state_ != State::kStaged) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        progress_.completed_blocks += prepared_blocks_;
        prepared_blocks_ = 0;
        is_complete = progress_.completed_blocks == progress_.logical_blocks;
        state_ = is_complete ? State::kComplete : State::kReady;
        if (should_trace_) {
            const auto saved_errno = errno;
            std::array<char, 160> message{};
            const auto written =
                std::snprintf(message.data(), message.size(),
                              "[glimmer] binary_slice blocks=%llu completed=%llu launches=%llu\n",
                              static_cast<unsigned long long>(progress_.logical_blocks),
                              static_cast<unsigned long long>(progress_.completed_blocks),
                              static_cast<unsigned long long>(progress_.physical_launches));
            if (written > 0 && static_cast<std::size_t>(written) < message.size())
                report_diagnostic(message.data());
            errno = saved_errno;
        }
        return CUDA_SUCCESS;
    }
    CudaBinarySliceProgress progress() const noexcept override {
        return progress_;
    }

   private:
    enum class State { kReady, kPrepared, kStaged, kComplete };
    std::shared_ptr<void> resources_;
    CudaBinarySliceProgress progress_;
    std::uint64_t width_;
    std::uint64_t prepared_blocks_{};
    State state_{State::kReady};
    bool should_trace_{};
};

}  // namespace

std::optional<CudaQueuedKernel> prepare_mapped_gemm_slices(CUfunction qualified_function,
                                                           std::array<unsigned int, 3> grid,
                                                           std::array<unsigned int, 3> block,
                                                           void* const* arguments,
                                                           std::shared_ptr<void> kernel_resources,
                                                           bool should_trace,
                                                           QualifiedGemmMapping mapping) noexcept {
    if (std::endian::native != std::endian::little || !qualified_function || !kernel_resources ||
        block != std::array<unsigned int, 3>{256, 1, 1} ||
        (mapping != QualifiedGemmMapping::kPlane && mapping != QualifiedGemmMapping::kRange))
        return std::nullopt;
    constexpr std::array<PtxParameterLayout, 1> original_layout{{{176, 8}}};
    auto original = OwnedKernelArguments::copy(original_layout, arguments);
    if (!original) return std::nullopt;
    GemmArguments parameters(original->addresses()[0]);
    if (!parameters.valid(grid)) return std::nullopt;
    constexpr std::array<PtxParameterLayout, 1> private_layout{{{184, 8}}};
    std::array<void*, 1> private_arguments{parameters.bytes.data()};
    auto owned = OwnedKernelArguments::copy(private_layout, private_arguments.data());
    if (!owned) return std::nullopt;
    try {
        auto continuation =
            std::make_shared<MappedGemmContinuation>(grid, kernel_resources, should_trace, mapping);
        return CudaQueuedKernel{
            qualified_function,     grid, block, 0, std::move(*owned), std::move(kernel_resources),
            std::move(continuation)};
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<CudaQueuedKernel> prepare_mapped_add_slices(CUfunction qualified_function,
                                                          std::array<unsigned int, 3> grid,
                                                          std::array<unsigned int, 3> block,
                                                          void* const* arguments,
                                                          std::shared_ptr<void> kernel_resources,
                                                          bool should_trace,
                                                          QualifiedAddMapping mapping) noexcept {
    if (std::endian::native != std::endian::little || !qualified_function || !kernel_resources ||
        grid[1] != 1 || grid[2] != 1 || block != std::array<unsigned int, 3>{128, 1, 1} ||
        (mapping != QualifiedAddMapping::kSingleCta && mapping != QualifiedAddMapping::kRange)) {
        return std::nullopt;
    }
    constexpr std::array<PtxParameterLayout, 3> original_layout{{{4, 4}, {4, 4}, {24, 8}}};
    auto original = OwnedKernelArguments::copy(original_layout, arguments);
    if (!original) {
        return std::nullopt;
    }
    std::int32_t count{};
    std::array<std::uint64_t, 4> aggregate{};
    std::memcpy(&count, original->addresses()[0], sizeof(count));
    std::memcpy(aggregate.data(), original->addresses()[2], 24);
    const auto maximum = mapping == QualifiedAddMapping::kRange ? 4194305 : 32769;
    if (count <= 0 || count > maximum ||
        grid[0] != (static_cast<unsigned int>(count) + 1023) / 1024) {
        return std::nullopt;
    }
    const auto bytes = static_cast<std::uint64_t>(count) * 4;
    for (std::size_t index = 0; index != 3; ++index) {
        if (!aggregate[index] || aggregate[index] % 16 != 0 ||
            aggregate[index] > std::numeric_limits<std::uint64_t>::max() - bytes) {
            return std::nullopt;
        }
    }
    for (std::size_t input = 1; input != 3; ++input) {
        if (aggregate[0] != aggregate[input] && aggregate[0] < aggregate[input] + bytes &&
            aggregate[input] < aggregate[0] + bytes) {
            return std::nullopt;
        }
    }
    constexpr std::array<PtxParameterLayout, 3> private_layout{{{4, 4}, {4, 4}, {32, 8}}};
    std::array<void*, 3> private_arguments{original->addresses()[0], original->addresses()[1],
                                           aggregate.data()};
    auto owned = OwnedKernelArguments::copy(private_layout, private_arguments.data());
    if (!owned) {
        return std::nullopt;
    }
    try {
        auto continuation = std::make_shared<MappedAddContinuation>(
            grid[0], kernel_resources, count <= 32769 ? 1 : 1024, should_trace);
        return CudaQueuedKernel{
            qualified_function,     grid, block, 0, std::move(*owned), std::move(kernel_resources),
            std::move(continuation)};
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace glimmer::interceptor
